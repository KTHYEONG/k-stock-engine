import logging
import os
import sys
from datetime import datetime
from config.base import LOG_DIR

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """시스템 로깅 설정 유틸리티"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    if log_file is None:
        log_file = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    # [CRITICAL] Windows 콘솔 인코딩 대응
    if sys.platform == "win32":
        try:
            # sys.stdout이 이미 utf-8이 아니면 재설정 시도
            if sys.stdout.encoding.lower() != 'utf-8':
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # File Handler - UTF-8 지정 (파일에는 항상 이모지 포함 저장)
    file_handler = logging.FileHandler(LOG_DIR / log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Stream Handler - stdout 사용
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 핸들러 중복 추가 방지
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger
