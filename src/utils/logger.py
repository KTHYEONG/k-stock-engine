import logging
import os
import sys
from datetime import datetime
from config.base import LOG_DIR

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """시스템 로깅 설정 유틸리티"""
    # [FIX] Windows 콘솔에서 이모지 출력을 위해 stdout 인코딩을 UTF-8로 강제
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
            
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    if log_file is None:
        log_file = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.FileHandler(LOG_DIR / log_file)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
