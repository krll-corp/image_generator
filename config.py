import os

class Config:
    """Configuration class for different deployment environments"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Server settings
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', '.')
    DEVICE = os.environ.get('DEVICE', 'cpu')  # 'cpu', 'cuda', 'mps'
    
    # Memory settings
    MAX_MEMORY_MB = int(os.environ.get('MAX_MEMORY_MB', 2048))
    
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    
class DockerConfig(ProductionConfig):
    MODEL_PATH = '/app/models'
    
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}