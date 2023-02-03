from web3 import Web3
from decouple import config
import json

# SETTINGS
SAVE_HISTORY = False
RUN_MODEL_1 = False
RUN_MODEL_2 = False
SCAN_OPPORTUNITIES = True

# Get ABI
with open("abi.json", "r") as myFile:
    data = myFile.read()
ABI = json.loads(data)

# Wallet details - mainnet
ACCOUNT_ADDRESS = config("ACCOUNT")
PRIVATE_KEY = config("PRIVATE_KEY")
PROVIDER_URL = config("PROVIDER")
CONTRACT_ADDRESS = config("CONTRACT_ADDRESS")

# Web 3 provider and contract
w3 = Web3(Web3.HTTPProvider(PROVIDER_URL))
CONTRACT = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

# Strategy
NUM_RECORDS_HISTORY = 3000
