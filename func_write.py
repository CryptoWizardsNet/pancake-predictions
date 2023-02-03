from web3 import Web3
from decouple import config


w3 = Web3(Web3.HTTPProvider(provider_url))
# Contract


# Send a Transaction
def send_tx(side):

  # Variables
  chain_id = 56
  gas = 300000
  gas_price = Web3.toWei("5.5", "gwei")
  send_bnb = 0.01
  amount = Web3.toWei(send_bnb, "ether")

  # Nonce
  nonce = w3.eth.getTransactionCount(account_address)

  # Build Transaction - BULL
  if side == "bull":
      tx_build = contract.functions.betBull(current_epoch).buildTransaction({
          "chainId": chain_id,
          "value": amount,
          "gas": gas,
          "gasPrice": gas_price,
          "nonce": nonce
      })

  if side == "bear":
      tx_build = contract.functions.betBear(current_epoch).buildTransaction({
          "chainId": chain_id,
          "value": amount,
          "gas": gas,
          "gasPrice": gas_price,
          "nonce": nonce
      })

  # Sign transaction
  tx_signed = w3.eth.account.signTransaction(tx_build, private_key=pk_mainnet)

  # Send transaction
  sent_tx = w3.eth.sendRawTransaction(tx_signed.rawTransaction)
  print(sent_tx)