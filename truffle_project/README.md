# VOKA AI PROTOCOL

## Installation
1. Requirement
	* NPM version 5.2 or later 
2. Setup
	* BLOCKCHAIN_PHRASE：12-word wallet mnemonic phrase 
	* infura corresponds to the links of each chain, such as polygon test chain mumbai, set MUMBAI_LINK
	* to run on an camp network, should set: camp_network
	* run `npm install`
4. Compiling
	* run `truffle compile`
6. Migrating
	* To migrate on camp network, run：`truffle deploy --network camp_network`

PS: Thee account corresponding to the mnemonic phrase needs to have currency on the chain to be deployed, which can be used to pay for gas.


## Introduction:
***VOKA AI Protocol*** is the embodiment of a new generation of Web 3.0 technologies. 
Not only are we capable of **generating exclusive anime avatars through our proprietary AI technology**; 
we can also use existing NFT PFPs to generate digital identities. 
Furthermore, we collaborate with some of the world's top designers and clothing brands 
to create a variety of virtual outfits and sell them through the Opensea platform. 
The application scenarios for your digital identities are boundless, including: 
video conferencing, webcasting, virtual social networking, and even web3 games.

**How to use:**
1.	Browse local website.
2.	Connect your wallet or create an account. 
3.	The protocol will scan the available NFT for generation in your wallet.
4.	If don’t have available NFT, you can use the Anime PFP Generation to create your own anime PFP.
5.	The protocol will generate 3D avatar from the selected anime display picture.
6.	After the generation, also can modify the assets of avatar, such as: clothes, hair style/color, eyeballs, and etc. 
7.	Not only the avatar can be saved on our server, we also support avatar mint. In order to ensure the avatar exclusive, the protocol will automatically remove the anime PFP from our system once the avatar mint.
8.	Our avatar standard format is suitable for the most of 3D engines (Unity or Unreal Engine) or Metaverse scenarios (Sandbox or Decentraland). In the coming future, we will provide SDK to import avatar file for metaverse creators and gaming developers. 

## Bonus:

**PC expression capture software:**

We also provide a UE5 PC expression capture software like Hologram for live-streaming and online chat. 
The software is support facial and upper limb capture.  

Before to use:
1. Since we don’t have enough time to create own UE live-streaming function, we use [3rd party UE plug-in](https://offworld.live/). 
2. In order to activate plug-in, you need to create an account on https://offworld.live/. 
3. Go to https://offworld.live/download-center, and click the “FREE DOWNLOAD” button.
Register an account with offworld.live, and click “Download” the plug-in. You can skip and cancel the download. 

  <img width="362" alt="image" src="https://user-images.githubusercontent.com/19359257/168949465-4a5c0757-7aa5-476b-93ca-0bba9b26e5aa.png">
  <img width="362" alt="image" src="https://user-images.githubusercontent.com/19359257/168949481-3a702918-7fb5-45b6-a8c0-86eb88e7bbdf.png">
  
4. Please contact us for the UE capture software.

**How to use**:

  TODO:...


## migration

1. set environment variables, `BLOCKCHAIN_PHRASE`, `INFURA_LINK`, etc.
2. migrate to specific blockchain, for instance `truffle migrate --network rinkeby` if we want migrate to testnet *rinkeby*.
3. To migrate on camp network, run: 'truffle migrate --network camp_network'
