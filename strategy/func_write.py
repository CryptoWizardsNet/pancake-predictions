from constants import w3, CONTRACT, ACCOUNT_ADDRESS, PRIVATE_KEY
from decouple import config
from web3 import Web3

# Send a Transaction
def send_tx(side):

  # Variables
  chain_id = 56
  gas = 300000
  gas_price = Web3.toWei("5.5", "gwei")
  send_bnb = config("WAGER_BNB")
  amount = Web3.toWei(send_bnb, "ether")

  # Get current epoch
  current_epoch = CONTRACT.functions.currentEpoch().call()

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


# Claim winnings
def claim_winnings():

  # Check is claimable
  current_epoch = CONTRACT.functions.currentEpoch().call()
  prev_epoch = current_epoch - 1

  # Claim winnings
  current_rounds_list = CONTRACT.functions.claimable(prev_epoch, ACCOUNT_ADDRESS).call()

  # Guard: No winnings to claim
  if not current_rounds_list:
    return False

  # Variables
  chain_id = 56
  gas = 300000
  gas_price = Web3.toWei("5.5", "gwei")

  # Nonce
  nonce = w3.eth.getTransactionCount(ACCOUNT_ADDRESS)

  # Caim Winnings
  tx_build = CONTRACT.functions.claim([prev_epoch]).buildTransaction({
      "chainId": chain_id,
      "gas": gas,
      "gasPrice": gas_price,
      "nonce": nonce
  })
  print(tx_build)

  # Sign transaction
  tx_signed = w3.eth.account.signTransaction(tx_build, private_key=PRIVATE_KEY)
  print(tx_signed)

  # Send transaction
  sent_tx = w3.eth.sendRawTransaction(tx_signed.rawTransaction)
  print(sent_tx)
