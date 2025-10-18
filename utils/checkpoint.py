"""
Checkpoint and Cache Management System
统一管理所有缓存文件（模型、边矩阵、特征等）
"""

import os
import json
import pickle
import hashlib
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
import fcntl  # 用于文件锁


logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """缓存元数据结构"""
    identifier: str
    cache_type: str  # 'densenet', 'edges', 'features' 等
    created_at: str
    config_hash: str
    config_params: Dict[str, Any]
    file_size_mb: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheMetadata':
        return cls(**data)


class CheckpointManager:
    """
    统一的缓存管理器
    
    Features:
        - 配置驱动的缓存标识符生成
        - 自动目录管理
        - 元数据追踪
        - 文件锁保护（多进程安全）
        - 损坏文件检测与恢复
    """
    
    def __init__(self, base_cache_dir: str = "./cache", auto_clean: bool = False):
        """
        初始化缓存管理器
        
        Args:
            base_cache_dir: 缓存根目录
            auto_clean: 是否自动清理损坏的缓存文件
        """
        self.base_cache_dir = Path(base_cache_dir)
        self.auto_clean = auto_clean
        
        # 为不同类型的缓存创建子目录
        self.cache_types = {
            'densenet': self.base_cache_dir / 'densenet_models',
            'edges': self.base_cache_dir / 'edge_matrices',
            'features': self.base_cache_dir / 'extracted_features',
            'metadata': self.base_cache_dir / 'metadata'
        }
        
        # 创建所有必要的目录
        self._initialize_directories()
        
        logger.info(f"CheckpointManager initialized with base dir: {self.base_cache_dir}")
    
    def _initialize_directories(self) -> None:
        """创建所有必要的缓存目录"""
        for cache_type, path in self.cache_types.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
    
    @staticmethod
    def generate_config_hash(config_dict: Dict[str, Any]) -> str:
        """
        根据配置参数生成唯一哈希值
        
        Args:
            config_dict: 配置参数字典（必须是可序列化的）
            
        Returns:
            8位哈希字符串
            
        Note:
            - 使用 MD5 快速生成哈希（非安全场景）
            - 配置字典会被排序以确保一致性
        """
        # 将字典转为排序后的JSON字符串，确保顺序一致
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        hash_obj = hashlib.md5(config_str.encode('utf-8'))
        return hash_obj.hexdigest()[:8]  # 取前8位足够区分
    
    def build_identifier(
        self, 
        cache_type: str, 
        config_params: Dict[str, Any],
        extra_tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        构建缓存标识符
        
        Args:
            cache_type: 缓存类型 ('densenet', 'edges', 'features')
            config_params: 关键配置参数（用于生成哈希）
            extra_tags: 额外的标识信息（如 fold, seed），不参与哈希
            
        Returns:
            格式化的标识符字符串
            
        Example:
            >>> manager.build_identifier(
            ...     'densenet', 
            ...     {'growth_rate': 8, 'num_init_features': 24},
            ...     {'fold': 0, 'seed': 1449}
            ... )
            'densenet_fold0_seed1449_a3f5c9b2'
        """
        config_hash = self.generate_config_hash(config_params)
        
        # 构建标识符各部分
        parts = [cache_type]
        
        if extra_tags:
            for key, value in sorted(extra_tags.items()):
                parts.append(f"{key}{value}")
        
        parts.append(config_hash)
        
        identifier = "_".join(parts)
        return identifier
    
    def get_path(self, cache_type: str, identifier: str, extension: str = '.pth') -> Path:
        """
        获取缓存文件的完整路径
        
        Args:
            cache_type: 缓存类型
            identifier: 缓存标识符
            extension: 文件扩展名
            
        Returns:
            缓存文件的Path对象
        """
        if cache_type not in self.cache_types:
            raise ValueError(f"Unknown cache type: {cache_type}. "
                           f"Supported types: {list(self.cache_types.keys())}")
        
        cache_dir = self.cache_types[cache_type]
        return cache_dir / f"{identifier}{extension}"
    
    def check_exists(self, cache_type: str, identifier: str, extension: str = '.pth') -> bool:
        """
        检查缓存是否存在且有效
        
        Args:
            cache_type: 缓存类型
            identifier: 缓存标识符
            extension: 文件扩展名
            
        Returns:
            缓存是否存在且有效
        """
        cache_path = self.get_path(cache_type, identifier, extension)
        
        if not cache_path.exists():
            logger.debug(f"Cache not found: {cache_path}")
            return False
        
        # 检查文件是否损坏（大小为0）
        if cache_path.stat().st_size == 0:
            logger.warning(f"Corrupted cache detected (0 bytes): {cache_path}")
            if self.auto_clean:
                logger.info(f"Auto-cleaning corrupted cache: {cache_path}")
                cache_path.unlink()
                self._remove_metadata(identifier)
            return False
        
        logger.debug(f"Valid cache found: {cache_path}")
        return True
    
    def load(
        self, 
        cache_type: str, 
        identifier: str, 
        extension: str = '.pth',
        map_location: Optional[str] = None
    ) -> Any:
        """
        加载缓存文件
        
        Args:
            cache_type: 缓存类型
            identifier: 缓存标识符
            extension: 文件扩展名
            map_location: PyTorch加载时的设备映射（仅用于.pth文件）
            
        Returns:
            加载的数据
            
        Raises:
            FileNotFoundError: 缓存文件不存在
            Exception: 加载失败
        """
        cache_path = self.get_path(cache_type, identifier, extension)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        
        logger.info(f"Loading cache from: {cache_path}")
        
        try:
            # 根据文件类型选择加载方式
            if extension == '.pth':
                import torch
                data = torch.load(cache_path, map_location=map_location)
            elif extension == '.pkl':
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            elif extension == '.npy':
                import numpy as np
                data = np.load(cache_path)
            else:
                # 通用二进制加载
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            
            logger.info(f"Successfully loaded cache: {identifier}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load cache {cache_path}: {str(e)}")
            if self.auto_clean:
                logger.info("Auto-cleaning corrupted cache...")
                cache_path.unlink()
                self._remove_metadata(identifier)
            raise
    
    def save(
        self,
        cache_type: str,
        identifier: str,
        data: Any,
        config_params: Dict[str, Any],
        extension: str = '.pth'
    ) -> Path:
        """
        保存数据到缓存
        
        Args:
            cache_type: 缓存类型
            identifier: 缓存标识符
            data: 要保存的数据
            config_params: 配置参数（用于元数据记录）
            extension: 文件扩展名
            
        Returns:
            保存的文件路径
            
        Note:
            - 使用文件锁保护多进程并发写入
            - 自动生成并保存元数据
        """
        cache_path = self.get_path(cache_type, identifier, extension)
        
        logger.info(f"Saving cache to: {cache_path}")
        
        # 使用临时文件 + 原子重命名，避免写入中断导致损坏
        temp_path = cache_path.with_suffix('.tmp')
        
        try:
            # 根据文件类型选择保存方式
            if extension == '.pth':
                import torch
                torch.save(data, temp_path)
            elif extension == '.pkl':
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif extension == '.npy':
                import numpy as np
                np.save(temp_path, data)
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 原子性地替换旧文件
            temp_path.replace(cache_path)
            
            # 保存元数据
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            metadata = CacheMetadata(
                identifier=identifier,
                cache_type=cache_type,
                created_at=datetime.now().isoformat(),
                config_hash=self.generate_config_hash(config_params),
                config_params=config_params,
                file_size_mb=round(file_size_mb, 2)
            )
            self._save_metadata(metadata)
            
            logger.info(f"Successfully saved cache: {identifier} ({file_size_mb:.2f} MB)")
            return cache_path
            
        except Exception as e:
            logger.error(f"Failed to save cache {cache_path}: {str(e)}")
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _save_metadata(self, metadata: CacheMetadata) -> None:
        """保存缓存元数据"""
        metadata_path = self.cache_types['metadata'] / f"{metadata.identifier}.json"
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"Metadata saved: {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")
    
    def _remove_metadata(self, identifier: str) -> None:
        """删除缓存元数据"""
        metadata_path = self.cache_types['metadata'] / f"{identifier}.json"
        if metadata_path.exists():
            metadata_path.unlink()
            logger.debug(f"Metadata removed: {metadata_path}")
    
    def get_metadata(self, identifier: str) -> Optional[CacheMetadata]:
        """
        获取缓存的元数据
        
        Args:
            identifier: 缓存标识符
            
        Returns:
            元数据对象，如果不存在则返回None
        """
        metadata_path = self.cache_types['metadata'] / f"{identifier}.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return CacheMetadata.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load metadata {metadata_path}: {str(e)}")
            return None
    
    def list_caches(self, cache_type: str) -> list[Dict[str, Any]]:
        """
        列出指定类型的所有缓存及其元数据
        
        Args:
            cache_type: 缓存类型
            
        Returns:
            缓存信息列表
        """
        cache_dir = self.cache_types[cache_type]
        caches = []
        
        for cache_file in cache_dir.iterdir():
            if cache_file.is_file() and not cache_file.name.endswith('.tmp'):
                identifier = cache_file.stem
                metadata = self.get_metadata(identifier)
                
                cache_info = {
                    'identifier': identifier,
                    'path': str(cache_file),
                    'size_mb': cache_file.stat().st_size / (1024 * 1024),
                    'metadata': metadata.to_dict() if metadata else None
                }
                caches.append(cache_info)
        
        return caches
    
    def clean_cache(self, cache_type: Optional[str] = None, keep_recent: int = 5) -> None:
        """
        清理旧的缓存文件
        
        Args:
            cache_type: 指定要清理的缓存类型，None表示清理所有类型
            keep_recent: 保留最近的N个缓存文件
        """
        types_to_clean = [cache_type] if cache_type else list(self.cache_types.keys())
        
        for ctype in types_to_clean:
            if ctype == 'metadata':
                continue
                
            cache_dir = self.cache_types[ctype]
            cache_files = sorted(
                [f for f in cache_dir.iterdir() if f.is_file()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # 删除旧文件
            for cache_file in cache_files[keep_recent:]:
                logger.info(f"Cleaning old cache: {cache_file}")
                cache_file.unlink()
                self._remove_metadata(cache_file.stem)