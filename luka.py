import uvicorn
import httpx
import asyncio
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Literal
import redis.asyncio as aioredis
from io import StringIO
from starlette.responses import JSONResponse


# --- Nuevas Importaciones de Iverson ---
import numpy as np

# Compatibilidad: algunas versiones de pandas_ta intentan importar `NaN` desde numpy
# (ej. `from numpy import NaN as npNaN`). Si la versión instalada de numpy no expone
# `NaN`, creamos un alias a `numpy.nan` para evitar ImportError.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import pandas_ta as ta
from xgboost import XGBRegressor

from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Modelos de Datos (Pydantic) ---

# Modelos existentes...
class Network(BaseModel):
    id: str
    type: str
    name: str

class DexTransactions(BaseModel):
    buys: Optional[int] = 0
    sells: Optional[int] = 0
    buyers: Optional[int] = 0
    sellers: Optional[int] = 0

class PoolAttributes(BaseModel):
    name: str
    address: str
    base_token_price_usd: str
    quote_token_price_usd: str
    price_change_percentage: Dict[str, str]
    volume_usd: Dict[str, str] = Field(..., alias='volume_usd')
    transactions: Optional[Dict[str, DexTransactions]] = Field(None, alias='transactions')

class Pool(BaseModel):
    id: str
    type: str
    attributes: PoolAttributes

class TradeAttributes(BaseModel):
    block_timestamp: str
    kind: str
    price_from_in_currency_token: str
    price_to_in_currency_token: str
    volume_in_usd: str

class Trade(BaseModel):
    id: str
    type: str
    attributes: TradeAttributes

class DexToken(BaseModel):
    address: str
    name: str
    symbol: str

class DexVolume(BaseModel):
    h24: Optional[float] = None
    h6: Optional[float] = None
    h1: Optional[float] = None
    m5: Optional[float] = None

class DexPriceChange(BaseModel):
    m5: Optional[float] = None
    h1: Optional[float] = None
    h6: Optional[float] = None
    h24: Optional[float] = None

class DexPair(BaseModel):
    chainId: str
    dexId: str
    url: str
    pairAddress: str
    baseToken: DexToken
    quoteToken: DexToken
    priceNative: str
    priceUsd: Optional[str] = None
    txns: Dict[str, DexTransactions]
    volume: DexVolume
    priceChange: DexPriceChange
    liquidity: Optional[Dict[str, float]] = None
    fdv: Optional[float] = None
    pairCreatedAt: Optional[int] = None

class Gem(BaseModel):
    symbol: str
    chain: str
    dex: str
    pair_address: str
    price_usd: float
    volume_24h: float
    price_change_24h: float
    liquidity_usd: float
    score: float
    reason: str

# --- Nuevos Modelos para Análisis Técnico ---
class TechnicalSignalResponse(BaseModel):
    signal: str
    price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: int # Añadido para el marcador del gráfico

class PredictionResponse(BaseModel):
    prediction_type: str
    current_price: float
    predicted_price: float
    confidence_score: Optional[float] = None
    details: str

class RiskAnalysisResponse(BaseModel):
    pair_address: str
    risk_score: float
    risk_level: str
    details: Dict[str, float]

class AnomalyAnalysisResponse(BaseModel):
    pair_address: str
    anomaly_score: float
    anomaly_level: str
    details: str


    
# --- Configuración Global ---
app = FastAPI(
    title="MCP Advanced Crypto Data API",
    description="Un servidor robusto para extraer y analizar datos de criptomonedas en tiempo real desde GeckoTerminal, DexScreener e Iverson.",
    version="4.5.0", # Versión incrementada por mejora de resiliencia
)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {exc}"},
    )

# --- Configuración de CORS ---
# Permitir solicitudes desde el frontend que corre en el puerto 8001
origins = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GECKO_BASE_URL = "https://api.geckoterminal.com/api/v2"
DEX_BASE_URL = "https://api.dexscreener.com/latest"
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Cabeceras para simular un navegador y mejorar la fiabilidad
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Configuración de Redis
REDIS_URL = "redis://localhost"
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
CACHE_TTL = 300 # 5 minutos

# --- Mapeo de Redes Centralizado (Single Source of Truth) ---
NETWORK_MAPPING = {
    # Page 1
    'eth': 'eth', 'ethereum': 'eth',
    'bsc': 'bsc', 'bnb chain': 'bsc', 'binance-smart-chain': 'bsc', 'binance': 'bsc',
    'polygon_pos': 'polygon_pos', 'polygon pos': 'polygon_pos', 'polygon-pos': 'polygon_pos', 'matic': 'polygon_pos',
    'avax': 'avax', 'avalanche': 'avax',
    'movr': 'movr', 'moonriver': 'movr',
    'cro': 'cro', 'cronos': 'cro',
    'one': 'one', 'harmony': 'one', 'harmony-shard-0': 'one',
    'boba': 'boba', 'boba network': 'boba',
    'ftm': 'ftm', 'fantom': 'ftm',
    'bch': 'bch', 'smartbch': 'bch',
    'aurora': 'aurora',
    'metis': 'metis', 'metis-andromeda': 'metis',
    'arbitrum': 'arbitrum', 'arbitrum-one': 'arbitrum',
    'fuse': 'fuse',
    'kcc': 'kcc', 'kucoin community chain': 'kcc', 'kucoin-community-chain': 'kcc',
    'iotx': 'iotx',
    'celo': 'celo',
    'xdai': 'xdai', 'gnosis xdai': 'xdai', 'gnosis': 'xdai',
    'glmr': 'glmr', 'moonbeam': 'glmr',
    'optimism': 'optimism', 'optimistic-ethereum': 'optimism',
    'nrg': 'nrg', 'energi': 'nrg',
    'wan': 'wan', 'wanchain': 'wan',
    'ronin': 'ronin',
    'kai': 'kai', 'kardiachain': 'kai',
    'mtr': 'mtr', 'meter': 'mtr',
    'velas': 'velas',
    'sdn': 'sdn', 'shiden': 'sdn', 'shiden network': 'sdn',
    'tlos': 'tlos', 'telos': 'tlos',
    'oasis': 'oasis', 'oasis emerald': 'oasis',
    'astr': 'astr', 'astar': 'astr',
    'ela': 'ela', 'elastos': 'ela',
    'milkada': 'milkada', 'milkomeda cardano': 'milkada', 'milkomeda-cardano': 'milkada',
    'dfk': 'dfk', 'dfk chain': 'dfk', 'defi-kingdoms-blockchain': 'dfk',
    'evmos': 'evmos',
    'solana': 'solana', 'sol': 'solana',
    'cfx': 'cfx', 'conflux': 'cfx',
    'bttc': 'bttc', 'bittorrent': 'bttc',
    'sxn': 'sxn', 'sx network': 'sxn', 'sx-network': 'sxn',
    'xdc': 'xdc',
    'kaia': 'kaia', 'klay-token': 'kaia',
    'kava': 'kava',
    'bitgert': 'bitgert',
    'tombchain': 'tombchain',
    'dogechain': 'dogechain',
    'findora': 'findora',
    'thundercore': 'thundercore',
    'arbitrum_nova': 'arbitrum_nova', 'arbitrum nova': 'arbitrum_nova', 'arbitrum-nova': 'arbitrum_nova',
    'canto': 'canto',
    'ethereum_classic': 'ethereum_classic', 'ethereum classic': 'ethereum_classic',
    'step-network': 'step-network', 'step network': 'step-network',
    'ethw': 'ethw', 'ethereumpow': 'ethw',
    'godwoken': 'godwoken',
    'songbird': 'songbird',
    'tomochain': 'tomochain', 'viction': 'tomochain',
    'platon_network': 'platon_network', 'platon network': 'platon_network',
    'exosama': 'exosama',
    'oasys': 'oasys',
    'bitkub_chain': 'bitkub_chain', 'kub': 'bitkub_chain',
    'wemix': 'wemix', 'wemix-network': 'wemix',
    'flare': 'flare', 'flare-network': 'flare',
    'onus': 'onus',
    'aptos': 'aptos',
    'core': 'core',
    'filecoin': 'filecoin',
    'zksync': 'zksync',
    'loopnetwork': 'loopnetwork',
    'multivac': 'multivac',
    'polygon-zkevm': 'polygon-zkevm', 'polygon zkevm': 'polygon-zkevm',
    'eos-evm': 'eos-evm', 'eos evm': 'eos-evm',
    'ultron': 'ultron',
    'sui-network': 'sui-network', 'sui network': 'sui-network', 'sui': 'sui-network',
    'pulsechain': 'pulsechain',
    'enuls': 'enuls',
    'tenet': 'tenet',
    'rollux': 'rollux',
    'starknet-alpha': 'starknet-alpha', 'starknet': 'starknet-alpha',
    'mantle': 'mantle',
    'neon-evm': 'neon-evm', 'neon evm': 'neon-evm',
    'linea': 'linea',
    'base': 'base',
    'bitrock': 'bitrock',
    'opbnb': 'opbnb',
    'sei-network': 'sei-network', 'sei network': 'sei-network',
    'shibarium': 'shibarium',
    'manta-pacific': 'manta-pacific', 'manta pacific': 'manta-pacific',
    'sepolia-testnet': 'sepolia-testnet', 'sepolia testnet': 'sepolia-testnet',
    'hedera-hashgraph': 'hedera-hashgraph', 'hedera hashgraph': 'hedera-hashgraph', 'hashgraph': 'hedera-hashgraph', 'hedera': 'hedera-hashgraph',
    'shimmerevm': 'shimmerevm', 'shimmer_evm': 'shimmerevm',
    'beam': 'beam',
    'scroll': 'scroll',
    'lightlink-phoenix': 'lightlink-phoenix', 'lightlink phoenix': 'lightlink-phoenix', 'lightlink': 'lightlink-phoenix',
    'elysium': 'elysium',
    'ton': 'ton', 'the-open-network': 'ton',
    'mode': 'mode',
    'defimetachain': 'defimetachain', 'defichain-evm': 'defimetachain',
    'humanode': 'humanode',
    'mxc-zkevm': 'mxc-zkevm', 'moonchain': 'mxc-zkevm',
    'zkfair': 'zkfair',
    'alveychain': 'alveychain',
    'hypra-network': 'hypra-network', 'hypra network': 'hypra-network',
    # Page 2
    'blast-sepolia-testnet': 'blast-sepolia-testnet', 'blast sepolia (testnet)': 'blast-sepolia-testnet',
    'zetachain': 'zetachain',
    'oasis-sapphire': 'oasis-sapphire', 'oasis sapphire': 'oasis-sapphire',
    'merlin-chain': 'merlin-chain', 'merlin chain': 'merlin-chain',
    'xai': 'xai',
    'immutable-zkevm': 'immutable-zkevm', 'immutable zkevm': 'immutable-zkevm', 'immutable': 'immutable-zkevm',
    'rails-network': 'rails-network', 'rails network': 'rails-network',
    'blast': 'blast',
    'areon-network': 'areon-network', 'areum network': 'areon-network',
    'map-protocol': 'map-protocol', 'map protocol': 'map-protocol',
    'fraxtal': 'fraxtal',
    'omax-chain': 'omax-chain', 'omax chain': 'omax-chain', 'omax': 'omax-chain',
    'zora-network': 'zora-network', 'zora': 'zora-network',
    'inevm': 'inevm',
    'bvm-nakachain': 'bvm-nakachain', 'bvm nakachain': 'bvm-nakachain',
    'graphlinq-chain': 'graphlinq-chain', 'graphlinq chain': 'graphlinq-chain',
    'qitmeer-network': 'qitmeer-network', 'qitmeer network': 'qitmeer-network',
    'bevm': 'bevm',
    'rss3-vsl-mainnet': 'rss3-vsl-mainnet', 'rss3 vsl mainnet': 'rss3-vsl-mainnet', 'rss3-vsl': 'rss3-vsl-mainnet',
    'degenchain': 'degenchain', 'degen chain': 'degenchain', 'degen': 'degenchain',
    'bahamut-mainnet': 'bahamut-mainnet', 'bahamut mainnet': 'bahamut-mainnet', 'bahamut': 'bahamut-mainnet',
    'chiliz-chain': 'chiliz-chain', 'chiliz chain': 'chiliz-chain', 'chiliz': 'chiliz-chain',
    'lukso': 'lukso',
    'ancient8': 'ancient8',
    'x-layer': 'x-layer', 'x layer': 'x-layer',
    'bsquared-network': 'bsquared-network', 'bsquared network': 'bsquared-network',
    'bitlayer': 'bitlayer',
    'bob-network': 'bob-network', 'bob network': 'bob-network',
    'redstone': 'redstone',
    'cyber': 'cyber',
    'octaspace': 'octaspace',
    'zklink-nova': 'zklink-nova', 'zklink nova': 'zklink-nova',
    'bouncebit': 'bouncebit',
    're-al': 're-al', 're.al': 're-al',
    'zedxion-smart-chain': 'zedxion-smart-chain', 'zedxion smart chain': 'zedxion-smart-chain', 'zedxion': 'zedxion-smart-chain',
    'genesys-network': 'genesys-network', 'genesys network': 'genesys-network',
    'taiko': 'taiko',
    'sanko-mainnet': 'sanko-mainnet', 'sanko': 'sanko-mainnet',
    'onchain': 'onchain',
    'jib-chain': 'jib-chain', 'jb chain': 'jib-chain', 'jibchain': 'jib-chain',
    'sei-evm': 'sei-evm', 'sei v2': 'sei-evm', 'sei-v2': 'sei-evm',
    'saakuru-mainnet': 'saakuru-mainnet', 'saakuru': 'saakuru-mainnet',
    'larissa-mainnet': 'larissa-mainnet', 'larissa': 'larissa-mainnet',
    'boba-bnb': 'boba-bnb', 'boba bnb': 'boba-bnb',
    'haqq-network': 'haqq-network', 'haqq network': 'haqq-network',
    'ham': 'ham',
    'rootstock': 'rootstock',
    'endurance': 'endurance',
    'iota-evm': 'iota-evm', 'iota evm': 'iota-evm',
    'alienx': 'alienx',
    'etherlink': 'etherlink',
    'skale-europa': 'skale-europa', 'skale europa': 'skale-europa', 'skale': 'skale-europa',
    'rari': 'rari',
    'nahmii': 'nahmii',
    'bomechain': 'bomechain',
    'zircuit': 'zircuit',
    'cronos-zkevm': 'cronos-zkevm', 'cronos zkevm': 'cronos-zkevm',
    'q-mainnet': 'q-mainnet', 'q mainnet': 'q-mainnet',
    'gravity-alpha': 'gravity-alpha', 'gravity': 'gravity-alpha',
    'tron': 'tron',
    'mint': 'mint',
    'swanchain': 'swanchain',
    'flow-evm': 'flow-evm', 'flow evm': 'flow-evm',
    'canxium': 'canxium',
    'shape': 'shape',
    'world-chain': 'world-chain', 'world chain': 'world-chain',
    'apechain': 'apechain',
    'morph-l2': 'morph-l2', 'morph l2': 'morph-l2',
    'cardano': 'cardano',
    'icp': 'icp', 'internet computer': 'icp', 'internet-computer': 'icp',
    'defiverse': 'defiverse',
    'laika': 'laika', 'laikachain': 'laika',
    'planq': 'planq', 'planq-network': 'planq',
    'zero-network': 'zero-network', 'zero network': 'zero-network',
    'ql1': 'ql1',
    'duckchain': 'duckchain',
    'vanarchain': 'vanarchain', 'vanar chain': 'vanarchain',
    'eclipse': 'eclipse',
    'airdao': 'airdao',
    'units-network': 'units-network', 'units network': 'units-network',
    'zilliqa-evm': 'zilliqa-evm', 'zilliqa evm': 'zilliqa-evm',
    'sonic': 'sonic',
    'ink': 'ink',
    'vana': 'vana',
    'electroneum': 'electroneum',
    'funki': 'funki',
    'matchain': 'matchain',
    'swellchain': 'swellchain',
    'soneium': 'soneium',
    'lisk': 'lisk',
    'educhain': 'educhain', 'edu chain': 'educhain',
    'xrpl': 'xrpl', 'xrp': 'xrpl',
    'ao': 'ao',
    'shido-network': 'shido-network', 'shido network': 'shido-network', 'shido': 'shido-network',
    'abstract': 'abstract',
    'artela': 'artela',
    'parex': 'parex', 'parex-network': 'parex',
    'berachain': 'berachain',
    'treasure': 'treasure',
    'taraxa': 'taraxa',
    # Page 3
    'unichain': 'unichain',
    'story': 'story',
    'hela': 'hela',
    'hyperevm': 'hyperevm',
    'monad-testnet': 'monad-testnet', 'monad testnet': 'monad-testnet',
    'corn': 'corn',
    'hyperliquid': 'hyperliquid',
    'pundi-aifx-omnilayer': 'pundi-aifx-omnilayer', 'pundi aifx omnilayer': 'pundi-aifx-omnilayer',
    'movement': 'movement',
    'form-network': 'form-network', 'form network': 'form-network',
    'saga': 'saga',
    'superseed': 'superseed',
    'shine-chain': 'shine-chain', 'shine chain': 'shine-chain',
    'sx-rollup': 'sx-rollup', 'sx rollup': 'sx-rollup',
    'goat': 'goat',
    'sonic-svm': 'sonic-svm', 'sonic svm': 'sonic-svm',
    'superposition': 'superposition',
    'lens': 'lens',
    'plume-network': 'plume-network', 'plume network': 'plume-network',
    'hemi': 'hemi',
    'sophon': 'sophon',
    'bittensor': 'bittensor',
    'tokchain': 'tokchain',
    'mantra': 'mantra',
    'bitcichain': 'bitcichain',
    'haven1': 'haven1',
    'initia': 'initia',
    'pepe-unchained': 'pepe-unchained', 'pepe unchained': 'pepe-unchained',
    'wax': 'wax',
    'hydra-chain': 'hydra-chain', 'hydra chain': 'hydra-chain',
    'katana': 'katana',
    'opengpu': 'opengpu',
    'redbelly-network': 'redbelly-network', 'redbelly network': 'redbelly-network',
    'botanix': 'botanix',
    'quai-network': 'quai-network', 'quai network': 'quai-network',
    'glue': 'glue',
    'near': 'near',
    'xrpl-evm': 'xrpl-evm', 'xrpl evm': 'xrpl-evm',
    'tac': 'tac',
    'memecore': 'memecore',
    'mezo': 'mezo',
    'hashkey': 'hashkey',
    'peaq': 'peaq',
    'camp-network': 'camp-network', 'camp network': 'camp-network',
    'somnia': 'somnia',
    'besc-hyperchain': 'besc-hyperchain', 'besc hyperchain': 'besc-hyperchain',
    'mitosis': 'mitosis',
    'nibiru': 'nibiru',
    'xone': 'xone',
    'plasma': 'plasma',
    '0g': '0g',
    'juchain': 'juchain',
    'gate-layer': 'gate-layer', 'gate layer': 'gate-layer',
}

def get_gecko_network_id(network_name: str) -> str:
    """
    Obtiene el ID de red de GeckoTerminal a partir de un nombre amigable de forma robusta.
    Es insensible a mayúsculas/minúsculas y busca en los alias.
    Lanza una excepción HTTPException si la red no es válida.
    """
    normalized_name = network_name.lower()
    gecko_id = NETWORK_MAPPING.get(normalized_name)
    
    if not gecko_id:
        # Si no se encuentra, mostrar una lista limpia de redes soportadas
        supported_networks = sorted(list(set(NETWORK_MAPPING.keys())))
        raise HTTPException(
            status_code=400,
            detail=f"Red '{network_name}' no soportada o no mapeada. Pruebe con uno de los siguientes: {supported_networks}"
        )
    return gecko_id

# --- Funciones Auxiliares para Análisis Técnico ---

async def _get_gecko_ohlcv(network_id: str, pool_address: str, timeframe: str = "1h", limit: int = 100, use_cache: bool = True) -> pd.DataFrame:
    """
    Obtiene datos OHLCV de GeckoTerminal, utilizando un caché en Redis y manejo de errores mejorado.
    """
    cache_key = f"ohlcv:{network_id}:{pool_address}:{timeframe}:{limit}"
    if use_cache:
        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache HIT para {cache_key}")
                return pd.read_json(StringIO(cached_data), orient='split')
        except Exception as e:
            logger.warning(f"Error al leer de Redis cache: {e}")

    logger.info(f"Cache MISS para {cache_key} (o caché omitido). Obteniendo datos de la API.")
    timeframe_map = {
        "5m": {"timeframe": "minute", "aggregate": 5},
        "15m": {"timeframe": "minute", "aggregate": 15},
        "1h": {"timeframe": "hour", "aggregate": 1},
        "4h": {"timeframe": "hour", "aggregate": 4},
        "12h": {"timeframe": "hour", "aggregate": 12},
        "1d": {"timeframe": "day", "aggregate": 1},
    }

    if timeframe not in timeframe_map:
        raise HTTPException(
            status_code=400, 
            detail=f"Timeframe no válido. Use uno de: {list(timeframe_map.keys())}"
        )

    gecko_params = timeframe_map[timeframe]
    url = f"{GECKO_BASE_URL}/networks/{network_id}/pools/{pool_address}/ohlcv/{gecko_params['timeframe']}"
    
    params = {"limit": limit, "aggregate": gecko_params['aggregate']}
    logger.info(f"Calling GeckoTerminal OHLCV API: {url} with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=HTTP_HEADERS)
            response.raise_for_status()
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                if "Challenge Verification" in response.text:
                    logger.error("Bloqueo de Cloudflare detectado en GeckoTerminal.")
                    raise HTTPException(status_code=503, detail="Servicio no disponible por verificación de seguridad de GeckoTerminal.")
                else:
                    logger.error(f"Respuesta no JSON inesperada de GeckoTerminal: {response.text[:200]}")
                    raise HTTPException(status_code=500, detail="Respuesta inesperada de la API de GeckoTerminal.")

            if not data.get('data') or not data['data'].get('attributes') or 'ohlcv_list' not in data['data']['attributes']:
                raise KeyError("La respuesta de la API no contiene 'ohlcv_list'.")

            ohlcv_data = data['data']['attributes']['ohlcv_list']
            if not ohlcv_data:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)

            try:
                df_json = df.to_json(orient='split')
                await redis_client.setex(cache_key, CACHE_TTL, df_json)
            except Exception as e:
                logger.warning(f"Error al escribir en Redis cache: {e}")

            return df

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"Datos OHLCV no encontrados para {pool_address} en {network_id}. URL: {url} - Error: {e.response.text}")
            return pd.DataFrame()
        else:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en GeckoTerminal: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con GeckoTerminal: {e}")
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Error procesando los datos de GeckoTerminal: {e}")

def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los indicadores técnicos (SMA, ATR y Volatilidad) para un DataFrame dado.
    """
    if df.empty:
        return df

    # Asegurarse de que las columnas necesarias existan
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame no contiene todas las columnas requeridas para indicadores: {required_columns}")

    df['MA4'] = ta.sma(df['close'], length=4)
    df['MA9'] = ta.sma(df['close'], length=9)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['volatilidad'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # Eliminar filas con NaN resultantes del cálculo de indicadores
    return df.dropna()

# --- Clases y Funciones de Predicción ---

class XGBoostPredictor:
    """Maneja el ciclo de vida completo de los modelos XGBoost."""
    def __init__(self, training_threshold=150, retrain_interval=144):
        self.models = {}
        self.training_threshold = training_threshold
        self.retrain_interval = retrain_interval

    def _model_path(self, symbol: str) -> str:
        return f"{MODEL_DIR}/{symbol.replace('/', '_')}_model.json"

    def _should_retrain(self, symbol: str, data_length: int) -> bool:
        model_exists = os.path.exists(self._model_path(symbol))
        return (not model_exists) or (data_length % self.retrain_interval == 0)

    def train_model(self, symbol: str, features: np.ndarray, target: np.ndarray) -> XGBRegressor:
        """Entrena y guarda un nuevo modelo con validación integrada."""
        try:
            if len(features) < self.training_threshold:
                return None
            
            split = int(len(features) * 0.8)
            X_train, X_val = features[:split], features[split:]
            y_train, y_val = target[:split], target[split:]
            
            model = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                early_stopping_rounds=15,
                objective='reg:squarederror'
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            model.save_model(self._model_path(symbol))
            return model
        except Exception as e:
            logger.error(f"Error entrenando modelo {symbol}: {e}")
            return None

    def get_model(self, symbol: str, features: np.ndarray, target: np.ndarray) -> XGBRegressor:
        """Obtiene el modelo existente o entrena uno nuevo."""
        try:
            if self._should_retrain(symbol, len(features)):
                return self.train_model(symbol, features, target)
            
            if symbol not in self.models:
                model = XGBRegressor()
                model.load_model(self._model_path(symbol))
                self.models[symbol] = model
            
            return self.models[symbol]
        except Exception as e:
            logger.error(f"Error cargando modelo {symbol}: {e}")
            return None

    def predict(self, df: pd.DataFrame) -> float:
        """Realiza una predicción de precio usando XGBoost."""
        features = df[['MA4', 'MA9', 'ATR']].values
        if len(features) < 2:
            return df['close'].iloc[-1]
        
        model = XGBRegressor()
        try:
            X_train, X_test = features[:-1], features[-1].reshape(1, -1)
            y_train = df['close'].values[1:]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            return float(pred)
        except Exception as e:
            logger.error(f"Error en predicción XGBoost: {e}")
            return df['close'].iloc[-1]

def monte_carlo_prediction(df: pd.DataFrame, simulations: int = 10000, days: int = 10) -> float:
    """Realiza una predicción de precio utilizando simulación Monte Carlo."""
    try:
        if df.empty or 'close' not in df.columns or 'volatilidad' not in df.columns:
            raise ValueError("DataFrame no válido o faltan columnas.")

        current_price = df['close'].iloc[-1]
        volatility = df['volatilidad'].iloc[-1]
        if current_price <= 0 or volatility <= 0 or np.isnan(volatility):
            return current_price # Retornar precio actual si no hay volatilidad

        daily_return = volatility / np.sqrt(252)
        price_paths = np.zeros((simulations, days))
        price_paths[:, 0] = current_price

        for day in range(1, days):
            rand_returns = np.random.normal(0, daily_return, simulations)
            price_paths[:, day] = price_paths[:, day - 1] * np.exp(rand_returns)

        predicted_price = np.median(price_paths[:, -1])
        
        ma4 = df['MA4'].iloc[-1]
        ma9 = df['MA9'].iloc[-1]
        if ma4 > ma9:
            predicted_price *= 1.02
        elif ma4 < ma9:
            predicted_price *= 0.98

        return float(predicted_price)
    except Exception as e:
        logger.error(f"Error en predicción Monte Carlo: {e}")
        return df['close'].iloc[-1]

def calculate_risk_score(pool_data: Pool) -> Dict:
    """Calcula un score de riesgo comprensivo para un pool."""
    risk_factors = {}
    attributes = pool_data.attributes

    # 1. Riesgo de Edad (más nuevo = más riesgoso)
    # Esta métrica no está directamente disponible en el objeto Pool de GeckoTerminal
    # Se podría estimar si tuviéramos la fecha de creación del pool.
    # Por ahora, lo omitimos o asignamos un riesgo neutral.
    risk_factors['age_risk'] = 0.2 # Neutral

    # 2. Riesgo de Liquidez
    liquidity = float(attributes.base_token_price_usd) * float(attributes.volume_usd.get('h24', 0))
    liquidity_risk = 0
    if liquidity < 10000:
        liquidity_risk = 0.8
    elif liquidity > 10000000:
        liquidity_risk = 0.4  # Riesgo de manipulación
    risk_factors['liquidity_risk'] = liquidity_risk * 0.3

    # 3. Riesgo de Relación Volumen/Liquidez
    vol_liq_ratio = float(attributes.volume_usd.get('h24', 0)) / max(liquidity, 1)
    if vol_liq_ratio > 5:  # Alta rotación
        risk_factors['vol_liq_ratio_risk'] = 0.6 * 0.3
    elif vol_liq_ratio < 0.1:  # Baja actividad
        risk_factors['vol_liq_ratio_risk'] = 0.4 * 0.3
    else:
        risk_factors['vol_liq_ratio_risk'] = 0.1 * 0.3

    # 4. Riesgo de Volatilidad de Precio
    price_change_24h = float(attributes.price_change_percentage.get('h24', 0))
    volatility_risk = min(abs(price_change_24h) / 50, 1.0)  # 50% de cambio = riesgo máximo
    risk_factors['volatility_risk'] = volatility_risk * 0.4

    total_risk_score = sum(risk_factors.values())
    return {"score": total_risk_score, "details": risk_factors}


# Instancia del predictor
xgb_predictor = XGBoostPredictor()

# --- Endpoints de Predicción ---

@app.get("/predict/xgboost/{network}/{pool_address}", response_model=PredictionResponse, tags=["Iverson - Predictions"])
async def get_xgboost_prediction(network: str, pool_address: str, timeframe: str = "1h"):
    """Predice el precio futuro de un token usando un modelo XGBoost."""
    try:
        gecko_network_id = get_gecko_network_id(network)

        # Ejecutar llamadas en paralelo
        ohlcv_task = _get_gecko_ohlcv(gecko_network_id, pool_address, timeframe=timeframe, limit=200, use_cache=False)
        pool_details_task = get_pool_details(network, pool_address)
        df, pool_data = await asyncio.gather(ohlcv_task, pool_details_task)

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail="Datos históricos insuficientes para la predicción.")
        if not pool_data:
            raise HTTPException(status_code=404, detail="No se pudieron obtener los detalles del pool en tiempo real.")

        # Usar el precio en tiempo real para la respuesta
        current_price = float(pool_data.attributes.base_token_price_usd)

        df_indicators = _calculate_technical_indicators(df)
        if df_indicators.empty:
            raise HTTPException(status_code=404, detail="No se pudieron calcular los indicadores técnicos para la predicción.")

        predicted_price = xgb_predictor.predict(df_indicators)

        return PredictionResponse(
            prediction_type="XGBoost",
            current_price=current_price, # Precio en tiempo real
            predicted_price=predicted_price,
            details="Predicción basada en indicadores técnicos históricos (SMA, ATR)."
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en predicción XGBoost: {e}")


@app.get("/predict/montecarlo/{network}/{pool_address}", response_model=PredictionResponse, tags=["Iverson - Predictions"])
async def get_montecarlo_prediction(network: str, pool_address: str, timeframe: str = "1h", simulations: int = 10000, days: int = 10):
    """Estima el precio futuro de un token usando Simulación de Monte Carlo."""
    try:
        gecko_network_id = get_gecko_network_id(network)

        # Ejecutar llamadas en paralelo
        ohlcv_task = _get_gecko_ohlcv(gecko_network_id, pool_address, timeframe=timeframe, limit=200, use_cache=False)
        pool_details_task = get_pool_details(network, pool_address)
        df, pool_data = await asyncio.gather(ohlcv_task, pool_details_task)

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail="Datos históricos insuficientes para la simulación.")
        if not pool_data:
            raise HTTPException(status_code=404, detail="No se pudieron obtener los detalles del pool en tiempo real.")

        # Usar el precio en tiempo real para la respuesta
        current_price = float(pool_data.attributes.base_token_price_usd)

        df_indicators = _calculate_technical_indicators(df)
        if df_indicators.empty or 'volatilidad' not in df_indicators.columns:
            raise HTTPException(status_code=404, detail="No se pudo calcular la volatilidad para la simulación.")

        # La simulación se basa en la volatilidad histórica, pero el precio de partida es el actual
        predicted_price = monte_carlo_prediction(df_indicators, simulations, days)

        return PredictionResponse(
            prediction_type="Monte Carlo",
            current_price=current_price, # Precio en tiempo real
            predicted_price=predicted_price,
            details=f"Estimación basada en {simulations} simulaciones sobre {days} días."
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en simulación Monte Carlo: {e}")

@app.get("/analyze/risk/{network}/{pool_address}", response_model=RiskAnalysisResponse, tags=["Iverson - Analysis"])
async def get_risk_analysis(network: str, pool_address: str):
    """Analiza el riesgo de un pool de liquidez."""
    try:
        gecko_network_id = get_gecko_network_id(network)
        pool_data = await get_pool_details(gecko_network_id, pool_address)
        
        risk_info = calculate_risk_score(pool_data)
        risk_score = risk_info['score']
        
        if risk_score > 0.7:
            risk_level = "CRÍTICO"
        elif risk_score > 0.5:
            risk_level = "ALTO"
        elif risk_score > 0.3:
            risk_level = "MEDIO"
        else:
            risk_level = "BAJO"

        return RiskAnalysisResponse(
            pair_address=pool_address,
            risk_score=risk_score,
            risk_level=risk_level,
            details=risk_info['details']
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en análisis de riesgo: {e}")

def calculate_anomaly_score(trades: List[Trade]) -> Dict:
    """Calcula un score de anomalía basado en las transacciones recientes."""
    if len(trades) < 20:
        return {"score": 0, "details": "Datos insuficientes para detectar anomalías."}

    df = pd.DataFrame([t.attributes.dict() for t in trades])
    df['volume_in_usd'] = pd.to_numeric(df['volume_in_usd'], errors='coerce')
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
    df = df.dropna(subset=['volume_in_usd', 'block_timestamp'])

    # Anomalía de Volumen
    volume_mean = df['volume_in_usd'].mean()
    volume_std = df['volume_in_usd'].std()
    volume_anomaly_score = (df['volume_in_usd'].iloc[-1] - volume_mean) / volume_std if volume_std > 0 else 0

    # Anomalía de Frecuencia
    time_diffs = df['block_timestamp'].diff().dt.total_seconds().dropna()
    freq_mean = time_diffs.mean()
    freq_std = time_diffs.std()
    last_time_diff = time_diffs.iloc[-1] if not time_diffs.empty else freq_mean
    freq_anomaly_score = (freq_mean - last_time_diff) / freq_std if freq_std > 0 else 0

    total_anomaly_score = (volume_anomaly_score + freq_anomaly_score) / 2
    details = f"Anomalía de Volumen: {volume_anomaly_score:.2f}, Anomalía de Frecuencia: {freq_anomaly_score:.2f}"

    return {"score": total_anomaly_score, "details": details}

@app.get("/analyze/anomaly/{network}/{pool_address}", response_model=AnomalyAnalysisResponse, tags=["Iverson - Analysis"])
async def get_anomaly_analysis(network: str, pool_address: str):
    """Analiza la actividad de trading de un pool en busca de anomalías."""
    try:
        gecko_network_id = get_gecko_network_id(network)
        trades = await get_pool_trades(gecko_network_id, pool_address)
        anomaly_info = calculate_anomaly_score(trades)
        anomaly_score = anomaly_info['score']

        if anomaly_score > 3:
            anomaly_level = "ALTA"
        elif anomaly_score > 1.5:
            anomaly_level = "MEDIA"
        else:
            anomaly_level = "BAJA"

        return AnomalyAnalysisResponse(
            pair_address=pool_address,
            anomaly_score=anomaly_score,
            anomaly_level=anomaly_level,
            details=anomaly_info['details']
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en análisis de anomalías: {e}")



class OHLCVData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class OHLCVResponse(BaseModel):
    data: List[OHLCVData]

@app.get("/ohlcv/{network}/{pool_address}", response_model=OHLCVResponse, tags=["GeckoTerminal - OHLCV"])
async def get_pool_ohlcv(
    network: str,
    pool_address: str,
    timeframe: str = "1h",
    limit: int = 100
):
    """
    Obtiene datos históricos OHLCV (Open, High, Low, Close, Volume) para un pool específico
    desde GeckoTerminal.
    """
    try:
        gecko_network_id = get_gecko_network_id(network)

        df = await _get_gecko_ohlcv(gecko_network_id, pool_address, timeframe=timeframe, limit=limit)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron datos OHLCV para el pool y timeframe especificados."
            )
        
        # Convertir DataFrame a lista de diccionarios para la respuesta Pydantic
        ohlcv_list = df.reset_index().rename(columns={'index': 'timestamp'}).to_dict(orient='records')
        return OHLCVResponse(data=ohlcv_list)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al obtener OHLCV: {e}")


# (Aquí se mantienen todos los endpoints que ya tenías en gecko.py)

@app.get("/networks", response_model=List[Network], tags=["GeckoTerminal - General"])
async def get_networks():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GECKO_BASE_URL}/networks", headers=HTTP_HEADERS)
            response.raise_for_status()
            data = response.json()['data']
            return [{"id": n['id'], "type": n['type'], "name": n['attributes']['name']} for n in data]
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error de API externa: {e}")

@app.get("/pool/{network}/{pool_address}", response_model=Pool, tags=["GeckoTerminal - Pools"])
async def get_pool_details(network: str, pool_address: str):
    try:
        gecko_network_id = get_gecko_network_id(network)
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GECKO_BASE_URL}/networks/{gecko_network_id}/pools/{pool_address}", headers=HTTP_HEADERS)
            response.raise_for_status()
            return Pool(**response.json()['data'])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Pool con dirección '{pool_address}' no encontrado en la red '{network}'.")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en GeckoTerminal: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con GeckoTerminal: {e}")

@app.get("/trades/{network}/{pool_address}", response_model=List[Trade], tags=["GeckoTerminal - Trades"])
async def get_pool_trades(network: str, pool_address: str):
    try:
        gecko_network_id = get_gecko_network_id(network)
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GECKO_BASE_URL}/networks/{gecko_network_id}/pools/{pool_address}/trades", headers=HTTP_HEADERS)
            response.raise_for_status()
            return [Trade(**t) for t in response.json()['data']]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Trades para el pool '{pool_address}' no encontrados en la red '{network}'.")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en GeckoTerminal: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con GeckoTerminal: {e}")

@app.get("/pools_by_volume/{network}", response_model=List[Pool], tags=["GeckoTerminal - Pools"])
async def get_pools_by_volume(network: str):
    try:
        gecko_network_id = get_gecko_network_id(network)
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GECKO_BASE_URL}/networks/{gecko_network_id}/pools?include=volume_usd", headers=HTTP_HEADERS)
            response.raise_for_status()
            return response.json()['data']
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Red '{network}' no encontrada en GeckoTerminal.")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en GeckoTerminal: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con GeckoTerminal: {e}")

@app.get("/pools/search", response_model=List[Pool], tags=["GeckoTerminal - Pools"])
async def search_pools(
    query: str,
    network: Optional[str] = None,
    page: Optional[int] = None,
    include: Optional[str] = None
):
    """
    Busca pools en GeckoTerminal por una consulta (query) en una red específica.
    """
    try:
        gecko_network_id = get_gecko_network_id(network) if network else None
        params = {
            "query": query,
            "network": gecko_network_id,
            "page": page,
            "include": include
        }
        # Filtrar parámetros None
        query_params = {k: v for k, v in params.items() if v is not None}

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{GECKO_BASE_URL}/search/pools", params=query_params, headers=HTTP_HEADERS)
            response.raise_for_status()
            try:
                json_data = response.json()
            except json.JSONDecodeError:
                if "Challenge Verification" in response.text:
                    logger.error("Bloqueo de Cloudflare detectado en GeckoTerminal (Search Pools).")
                    raise HTTPException(status_code=503, detail="Servicio no disponible por verificación de seguridad de GeckoTerminal.")
                else:
                    logger.error(f"Respuesta no JSON inesperada de GeckoTerminal: {response.text[:200]}")
                    raise HTTPException(status_code=500, detail="Respuesta inesperada de la API de GeckoTerminal.")
            
            pools_data = []
            if isinstance(json_data.get('data'), list):
                for pool_item in json_data['data']:
                    if pool_item.get('type') == 'pool': # Ensure it's a pool
                        pools_data.append({
                            "id": pool_item["id"],
                            "type": pool_item["type"],
                            "attributes": {
                                "name": pool_item["attributes"]["name"],
                                "address": pool_item["attributes"]["address"],
                                "base_token_price_usd": pool_item["attributes"].get("base_token_price_usd", "0"),
                                "quote_token_price_usd": pool_item["attributes"].get("quote_token_price_usd", "0"),
                                "price_change_percentage": pool_item["attributes"].get("price_change_percentage", {}),
                                "volume_usd": pool_item["attributes"].get("volume_usd", {}),
                                "transactions": pool_item["attributes"].get("transactions", {})
                            }
                        })
            
            return [Pool(**pool_data) for pool_data in pools_data]

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en GeckoTerminal (Search Pools): {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con GeckoTerminal (Search Pools): {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al buscar pools: {e}")

@app.get("/pairs/search", response_model=List[DexPair], response_model_exclude_unset=False, tags=["DexScreener - Pares"])
async def search_dex_pairs_advanced(
    query: str,
    sort_by: Optional[str] = None,
    order: Optional[str] = 'desc'
):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{DEX_BASE_URL}/dex/search?q={query}", headers=HTTP_HEADERS)
            response.raise_for_status()
            json_data = response.json()
            pairs = json_data.get('pairs', [])

        if sort_by and pairs:
            reverse = order == 'desc'
            
            def get_sort_key(pair):
                try:
                    if sort_by == 'priceChange':
                        return float(pair.get('priceChange', {}).get('h24', 0) or 0)
                    elif sort_by == 'volume':
                        return float(pair.get('volume', {}).get('h24', 0) or 0)
                    elif sort_by == 'createdAt':
                        return int(pair.get('pairCreatedAt', 0) or 0)
                    return 0
                except (TypeError, ValueError, AttributeError):
                    return 0

            valid_sort_by = ['priceChange', 'volume', 'createdAt']
            if sort_by not in valid_sort_by:
                raise HTTPException(status_code=400, detail=f"Criterio 'sort_by' no válido. Use {valid_sort_by}")

            pairs.sort(key=get_sort_key, reverse=reverse)

        return pairs
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error de API externa: {e}")

@app.get("/pair/{chain_id}/{pair_address}", response_model=DexPair, tags=["DexScreener - Pares"])
async def get_dex_pair_details(chain_id: str, pair_address: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{DEX_BASE_URL}/dex/pairs/{chain_id}/{pair_address}", headers=HTTP_HEADERS)
            response.raise_for_status()
            pair_data = response.json().get('pair')
            if not pair_data:
                raise HTTPException(status_code=404, detail="Par no encontrado en DexScreener.")
            return pair_data
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Par con dirección '{pair_address}' no encontrado en la red '{chain_id}' en DexScreener.")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en DexScreener: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con DexScreener: {e}")

class PairTransactionsSummary(BaseModel):
    m5: Optional[DexTransactions] = None
    h1: Optional[DexTransactions] = None
    h6: Optional[DexTransactions] = None
    h24: Optional[DexTransactions] = None

@app.get("/pair/{chain_id}/{pair_address}/transactions", response_model=PairTransactionsSummary, tags=["DexScreener - Pares"])
async def get_dex_pair_transactions(chain_id: str, pair_address: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{DEX_BASE_URL}/dex/pairs/{chain_id}/{pair_address}", headers=HTTP_HEADERS)
            response.raise_for_status()
            pair_data = response.json().get('pair')
            if not pair_data:
                raise HTTPException(status_code=404, detail="Par no encontrado en DexScreener.")
            return pair_data.get('txns', {})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Par con dirección '{pair_address}' no encontrado en la red '{chain_id}' en DexScreener.")
        else:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en DexScreener: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con DexScreener: {e}")

# --- Endpoint de Señales Técnicas ---

@app.get("/signals/{network}/{pool_address}", response_model=TechnicalSignalResponse, tags=["Iverson - Technical Analysis"])
async def get_technical_signal(
    network: str,
    pool_address: str,
    timeframe: str = "4h"
):
    """
    Genera una señal de trading (Compra/Venta/Mantener) para un par de tokens específico
    basado en el estado de las medias móviles (SMA 4 y 9) y calcula los niveles de
    Stop-Loss y Take-Profit basados en el ATR (Average True Range) y el precio en tiempo real.
    """
    try:
        gecko_network_id = get_gecko_network_id(network)

        # 1. Ejecutar ambas llamadas de API en paralelo para máxima eficiencia
        ohlcv_task = _get_gecko_ohlcv(gecko_network_id, pool_address, timeframe=timeframe, limit=150, use_cache=False)
        pool_details_task = get_pool_details(network, pool_address)
        
        df, pool_data = await asyncio.gather(ohlcv_task, pool_details_task)

        # 2. Validar los datos de ambas fuentes
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No se pudieron obtener datos históricos OHLCV para el par y timeframe especificados."
            )
        if not pool_data:
            raise HTTPException(
                status_code=404,
                detail="No se pudieron obtener los detalles del pool en tiempo real."
            )

        # 3. Extraer los datos necesarios
        # Precio EN TIEMPO REAL de los detalles del pool
        current_price = float(pool_data.attributes.base_token_price_usd)

        # Indicadores de los datos HISTÓRICOS
        df_indicators = _calculate_technical_indicators(df)
        if len(df_indicators) < 2:
            raise HTTPException(
                status_code=400,
                detail="No hay suficientes datos históricos para calcular indicadores."
            )
        
        last_row = df_indicators.iloc[-1]
        atr_value = last_row.get('ATR')
        ma4 = last_row.get('MA4')
        ma9 = last_row.get('MA9')
        signal_timestamp = int(datetime.now().timestamp()) # Usar el tiempo actual para la señal

        # Validar que los datos numéricos esenciales no son nulos o cero
        if not all(isinstance(v, (int, float)) and v > 0 for v in [current_price, atr_value, ma4, ma9]):
             raise HTTPException(
                status_code=400,
                detail=f"Datos de indicadores o precio en tiempo real insuficientes o inválidos. Price: {current_price}, ATR: {atr_value}, MA4: {ma4}, MA9: {ma9}"
            )

        # 4. Lógica de señal basada en el estado de los indicadores históricos
        if ma4 > ma9:
            signal = "Compra"
            reason = f"Señal de Compra (Long): La media móvil rápida (MA4={ma4:.4f}) está por encima de la lenta (MA9={ma9:.4f})."
        elif ma4 < ma9:
            signal = "Venta"
            reason = f"Señal de Venta (Short): La media móvil rápida (MA4={ma4:.4f}) está por debajo de la lenta (MA9={ma9:.4f})."
        else:
            signal = "Mantener"
            reason = f"Señal Neutral: Las medias móviles (MA4 y MA9) están en el mismo valor ({ma4:.4f})."

        # 5. Calcular Stop-Loss y Take-Profit usando el PRECIO EN TIEMPO REAL
        stop_loss_multiplier = 1.5
        risk_reward_ratio = 1.7
        take_profit_multiplier = stop_loss_multiplier * risk_reward_ratio

        if signal == "Compra":
            stop_loss = current_price - (atr_value * stop_loss_multiplier)
            take_profit = current_price + (atr_value * take_profit_multiplier)
        elif signal == "Venta":
            stop_loss = current_price + (atr_value * stop_loss_multiplier)
            take_profit = current_price - (atr_value * take_profit_multiplier)
        else:  # Mantener
            stop_loss = current_price - (atr_value * stop_loss_multiplier)
            take_profit = current_price + (atr_value * take_profit_multiplier)
        
        # Formatear razón con precisión mejorada para evitar 0.00 en precios bajos
        # y añadir la relación riesgo/recompensa para mayor claridad.
        precision = 8 if current_price < 1 else 4
        
        # Redondear los valores finales antes de devolverlos
        stop_loss = round(max(0, stop_loss), precision)
        take_profit = round(max(0, take_profit), precision)

        if signal == "Mantener":
            reason += f" Precio actual: {current_price:.{precision}f}. Soporte estimado: {stop_loss:.{precision}f}, Resistencia estimada: {take_profit:.{precision}f}."
        else:
            reason += f" SL: {stop_loss:.{precision}f}, TP: {take_profit:.{precision}f} (Ratio R/R: {risk_reward_ratio})."

        return TechnicalSignalResponse(
            signal=signal,
            price=current_price, # Devolver el precio en tiempo real
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            timestamp=signal_timestamp
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado en get_technical_signal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al generar la señal: {e}")



@app.get("/pools/megafilter", response_model=List[Pool], tags=["GeckoTerminal - Pools"])
async def get_pools_megafilter(
    buy_tax_percentage_max: Optional[float] = None,
    buy_tax_percentage_min: Optional[float] = None,
    buys_duration: Optional[Literal['5m', '1h', '6h', '24h']] = None,
    buys_max: Optional[int] = None,
    buys_min: Optional[int] = None,
    checks: Optional[str] = None,
    dexes: Optional[str] = None,
    fdv_usd_max: Optional[float] = None,
    fdv_usd_min: Optional[float] = None,
    h24_volume_usd_max: Optional[float] = None,
    h24_volume_usd_min: Optional[float] = None,
    include: Optional[str] = None,
    networks: Optional[str] = None,
    page: Optional[int] = None,
    pool_created_hour_max: Optional[float] = None,
    pool_created_hour_min: Optional[float] = None,
    reserve_in_usd_max: Optional[float] = None,
    reserve_in_usd_min: Optional[float] = None,
    sell_tax_percentage_max: Optional[float] = None,
    sell_tax_percentage_min: Optional[float] = None,
    sells_duration: Optional[Literal['5m', '1h', '6h', '24h']] = None,
    sells_max: Optional[int] = None,
    sells_min: Optional[int] = None,
    sort: Optional[Literal['m5_trending', 'h1_trending', 'h6_trending', 'h24_trending', 'h24_tx_count_desc', 'h24_volume_usd_desc', 'h24_price_change_percentage_desc', 'pool_created_at_desc']] = None,
    tx_count_duration: Optional[Literal['5m', '1h', '6h', '24h']] = None,
    tx_count_max: Optional[int] = None,
    tx_count_min: Optional[int] = None,
):
    """
    Permite buscar pools en GeckoTerminal utilizando una amplia gama de filtros.
    """
    params = {k: v for k, v in locals().items() if v is not None and k not in ["self", "client"]}
    
    try:
        gecko_megafilter_url = f"{GECKO_BASE_URL}/pools/megafilter"
        query_params = {k: str(v) for k, v in params.items()}

        async with httpx.AsyncClient() as client:
            response = await client.get(gecko_megafilter_url, params=query_params, headers=HTTP_HEADERS)
            response.raise_for_status()
            json_data = response.json()
            pools_data = json_data.get('data', [])
            
            return [Pool(**pool_data) for pool_data in pools_data]

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en GeckoTerminal (Megafilter): {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con GeckoTerminal (Megafilter): {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al usar Megafilter: {e}")


# --- Motores de Descubrimiento ---
# (Se mantienen los motores de descubrimiento que ya tenías)
class BoostedOpportunity(BaseModel):
    symbol: str
    chain: str
    pair_address: str
    price_usd: float
    volume_24h: float
    price_change_24h: float
    buy_sell_ratio_24h: float
    buy_sell_ratio_6h: float 
    transactions_24h: DexTransactions
    boost_amount: int
    score: float
    reason: str

@app.get("/discover/boosted_opportunities", response_model=List[BoostedOpportunity], tags=["Discovery Engine"])
async def find_boosted_opportunities(
    min_volume: float = 20000,
    min_price_change: float = 10.0,
    min_buy_sell_ratio: float = 1.5
):
    async with httpx.AsyncClient() as client:
        try:
            # 1. Obtener tokens con "boost"
            boosts_response = await client.get("https://api.dexscreener.com/token-boosts/top/v1", headers=HTTP_HEADERS)
            boosts_response.raise_for_status()
            boosted_tokens = boosts_response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Error al obtener boosts de DexScreener: {e}")
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=500, detail=f"Formato de respuesta de boosts inesperado: {e}")

        # 2. Función auxiliar para obtener y procesar datos de un par
        async def fetch_pair_data(token_boost):
            try:
                token_address = token_boost.get('tokenAddress')
                if not token_address:
                    return None

                search_response = await client.get(f"{DEX_BASE_URL}/dex/search?q={token_address}", headers=HTTP_HEADERS)
                search_response.raise_for_status()
                pairs = search_response.json().get('pairs', [])
                
                if not pairs:
                    return None

                # Seleccionar el par con mayor volumen como el más representativo
                main_pair = max(pairs, key=lambda p: float(p.get('volume', {}).get('h24', 0) or 0))
                return main_pair, token_boost.get('amount', 0)
            except (httpx.RequestError, KeyError, ValueError):
                return None # Ignorar errores para un token individual

        # 3. Procesar todos los tokens en paralelo
        tasks = [fetch_pair_data(token) for token in boosted_tokens]
        results = await asyncio.gather(*tasks)

        # 4. Filtrar y calificar las oportunidades
        opportunities = []
        for result in filter(None, results):
            pair, boost_amount = result
            try:
                volume_24h = float(pair.get('volume', {}).get('h24', 0) or 0)
                price_change_24h = float(pair.get('priceChange', {}).get('h24', 0) or 0)
                txns_24h = pair.get('txns', {}).get('h24', {'buys': 0, 'sells': 0})
                buys = txns_24h.get('buys', 0)
                sells = txns_24h.get('sells', 0)
                buy_sell_ratio = buys / (sells + 1) # Evitar división por cero

                # Aplicar filtros definidos por el usuario
                if volume_24h < min_volume or price_change_24h < min_price_change or buy_sell_ratio < min_buy_sell_ratio:
                    continue

                # Calcular puntuación para ranking
                score = (volume_24h / 50000) + (price_change_24h * 1.5) + (buy_sell_ratio * 2) + (boost_amount / 1000)
                
                opportunities.append(BoostedOpportunity(
                    symbol=pair.get('baseToken', {}).get('symbol', 'N/A'),
                    chain=pair.get('chainId', 'N/A'),
                    pair_address=pair.get('pairAddress', 'N/A'),
                    price_usd=float(pair.get('priceUsd', 0) or 0),
                    volume_24h=volume_24h,
                    price_change_24h=price_change_24h,
                    buy_sell_ratio_24h=buy_sell_ratio,
                    transactions_24h=txns_24h,
                    boost_amount=boost_amount,
                    score=score,
                    reason="Fuerte presión de compra, alto volumen y cambio de precio significativo en un token promocionado."
                ))
            except (TypeError, ValueError, KeyError):
                continue # Ignorar si un par tiene datos mal formados
    
    return sorted(opportunities, key=lambda x: x.score, reverse=True)[:25]


@app.get("/discover/momentum_gems", response_model=List[Gem], tags=["Discovery Engine"])
async def find_momentum_gems(
    chain: str = 'ethereum', # Cambiado a ethereum por defecto para más resultados
    min_volume: float = 10000,
    min_liquidity: float = 5000,
    min_price_change: float = 5.0, # Umbral más bajo para encontrar más tokens
    duration: Literal['5m', '1h', '6h', '24h'] = '24h' # Duración para trending pools
):
    try:
        gecko_network_id = get_gecko_network_id(chain)
        async with httpx.AsyncClient() as client:
            # 1. Corregir la URL para incluir los datos necesarios
            request_url = f"{GECKO_BASE_URL}/networks/{gecko_network_id}/pools"
            params = {"include": "base_token,dex", "page": 1}
            
            logger.info(f"Calling GeckoTerminal Pools API: {request_url} with params: {params}")
            response = await client.get(request_url, params=params, headers=HTTP_HEADERS)
            response.raise_for_status()
            json_data = response.json()
            
            all_pools = json_data.get('data', [])
            included_data = json_data.get('included', [])

            # 2. Crear mapeos robustos para tokens y DEXes
            token_details = {}
            dex_details = {}
            for item in included_data:
                if isinstance(item, dict) and item.get('type') == 'token':
                    item_id = item.get('id')
                    if isinstance(item_id, str):
                        token_details[item_id] = item.get('attributes', {})
                elif isinstance(item, dict) and item.get('type') == 'dex':
                    item_id = item.get('id')
                    if isinstance(item_id, str):
                        dex_details[item_id] = item.get('attributes', {})

        potential_gems = []
        for pool in all_pools:
            try:
                attributes = pool.get('attributes', {})
                relationships = pool.get('relationships', {})
                
                base_token_id = relationships.get('base_token', {}).get('data', {}).get('id')
                dex_id = relationships.get('dex', {}).get('data', {}).get('id')

                # 3. Asegurarse de que tenemos la información antes de procesar
                if not base_token_id or not dex_id:
                    logger.warning(f"Skipping pool due to missing base_token_id or dex_id: {pool.get('id', 'N/A')}")
                    continue

                symbol = token_details.get(base_token_id, {}).get('symbol', 'N/A')
                dex_name = dex_details.get(dex_id, {}).get('name', 'N/A')
                
                if symbol == 'N/A' or dex_name == 'N/A':
                    logger.warning(f"Skipping pool due to missing symbol or dex_name for pool {pool.get('id', 'N/A')}. Base Token ID: {base_token_id}, Dex ID: {dex_id}")
                    continue

                pool_address = attributes.get('address', 'N/A')
                volume_24h = float(attributes.get('volume_usd', {}).get('h24', 0) or 0)
                liquidity_usd = float(attributes.get('reserve_in_usd', 0) or 0)
                price_change_24h = float(attributes.get('price_change_percentage', {}).get('h24', 0) or 0)
                price_usd = float(attributes.get('base_token_price_usd', 0) or 0)

                if (
                    volume_24h > min_volume and
                    liquidity_usd > min_liquidity and
                    price_change_24h > min_price_change
                ):
                    score = (volume_24h / 10000) + (price_change_24h * 2) + (liquidity_usd / 20000)
                    potential_gems.append(Gem(
                        symbol=symbol,
                        chain=chain,
                        dex=dex_name,
                        pair_address=pool_address,
                        price_usd=price_usd,
                        volume_24h=volume_24h,
                        price_change_24h=price_change_24h,
                        liquidity_usd=liquidity_usd,
                        score=score,
                        reason="High volume, liquidity, and positive price change from top pools."
                    ))
            except (TypeError, ValueError, KeyError) as e:
                logger.error(f"Error procesando pool de GeckoTerminal (Momentum Gems): {e} - Pool data: {pool}")
                continue

        return sorted(potential_gems, key=lambda x: x.score, reverse=True)[:20]

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error de API en GeckoTerminal (Momentum Gems): {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error de conexión con GeckoTerminal (Momentum Gems): {e}")
    except Exception as e:
        logger.error(f"Error inesperado en find_momentum_gems: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error inesperado procesando datos de GeckoTerminal: {str(e)}")

# --- Ejecución del Servidor ---

if __name__ == "__main__":
    print('Iniciando servidor Uvicorn en http://127.0.0.1:8000')
    print('Accede a la documentación interactiva en http://127.0.0.1:8000/docs')
    print('Para probar los endpoints, inicia el servidor y usa un cliente HTTP como curl o Postman.')
    
    uvicorn.run(app, host="127.0.0.1", port=8000)