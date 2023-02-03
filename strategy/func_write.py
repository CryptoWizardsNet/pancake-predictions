from constants import w3, CONTRACT, ACCOUNT_ADDRESS, PRIVATE_KEY
from func_read import get_current_epoch
from web3 import Web3

# Send a Transaction
def send_tx(side):

  # Variables
  chain_id = 56
  gas = 300000
  gas_price = Web3.toWei("5.5", "gwei")
  send_bnb = 0.01
  amount = Web3.toWei(send_bnb, "ether")

  # Get current epoch
  current_epoch = get_current_epoch()

  # Nonce
  nonce = w3.eth.getTransactionCount(ACCOUNT_ADDRESS)

  # Build Transaction - BULL
  if side == "bull":
    tx_build = CONTRACT.functions.betBull(current_epoch).buildTransaction({
      "chainId": chain_id,
      "value": amount,
      "gas": gas,
      "gasPrice": gas_price,
      "nonce": nonce
    })

  if side == "bear":
    tx_build = CONTRACT.functions.betBear(current_epoch).buildTransaction({
      "chainId": chain_id,
      "value": amount,
      "gas": gas,
      "gasPrice": gas_price,
      "nonce": nonce
    })

  # Sign transaction
  tx_signed = w3.eth.account.signTransaction(tx_build, private_key=PRIVATE_KEY)

  # Send transaction
  sent_tx = w3.eth.sendRawTransaction(tx_signed.rawTransaction)
  print(sent_tx)