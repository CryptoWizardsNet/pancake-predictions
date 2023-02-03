import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from datetime import datetime, timezone
from web3 import Web3
from decouple import config
import pandas as pd
import json
import time

# Contract address and ABI details
address_contract = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"






# Current Epoch/Round
current_epoch = contract.functions.currentEpoch().call()



def look_to_trade():

    # Current Rounds information
    current_rounds_list = contract.functions.rounds(current_epoch).call()
    lock_timestamp = current_rounds_list[2]
    total_amount = current_rounds_list[8]
    bull_amount = current_rounds_list[9]
    bear_amount = current_rounds_list[10]

    # Get current timestamp
    dt = int(datetime.now(timezone.utc).timestamp())
    time_remaining = lock_timestamp - dt

    # Calculate Ratio
    total_amount_normal = round(float(Web3.fromWei(total_amount, "ether")), 5)
    bull_amount_normal = round(float(Web3.fromWei(bull_amount, "ether")), 5)
    bear_amount_normal = round(float(Web3.fromWei(bear_amount, "ether")), 5)

    # Ratios
    if bull_amount_normal != 0 and bear_amount_normal != 0:
        bull_ratio = round(bull_amount_normal / bear_amount_normal, 2) + 1
        bear_ratio = round(bear_amount_normal / bull_amount_normal, 2) + 1
    else:
        bull_ratio = 0
        bear_ratio = 0

    # Ratios
    print(bull_ratio, bear_ratio)

    # Place trade
    if time_remaining <= 10:
        if bull_ratio > bear_ratio:
            send_tx("bull")
        else:
            send_tx("bear")
        print("done")


def claim_winnings(epoch):

    # Variables
    chain_id = 56
    gas = 300000
    gas_price = Web3.toWei("5.5", "gwei")

    # Nonce
    nonce = w3.eth.getTransactionCount(account_address_mainnet)

    # Caim Winnings
    tx_build = contract.functions.claim([epoch]).buildTransaction({
        "chainId": chain_id,
        "gas": gas,
        "gasPrice": gas_price,
        "nonce": nonce
    })
    print(tx_build)

    # Sign transaction
    tx_signed = w3.eth.account.signTransaction(tx_build, private_key=pk_mainnet)
    print(tx_signed)

    # Send transaction
    sent_tx = w3.eth.sendRawTransaction(tx_signed.rawTransaction)
    print(sent_tx)

claim_winnings(66562)
# look_to_trade()
