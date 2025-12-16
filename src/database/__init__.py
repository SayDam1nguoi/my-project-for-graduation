# -*- coding: utf-8 -*-
"""
Database Module

Quản lý kết nối và thao tác với database (MongoDB).
"""

from .mongodb_manager import MongoDBManager, get_mongodb_manager

__all__ = ['MongoDBManager', 'get_mongodb_manager']
