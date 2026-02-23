
Q-less solver by Indy and Emerson

**Plan**
- Pick letters randomly from the set of 12 dice
- Algorithm to take in letters and generate all possible starting words
- Then use some sort of DFS to search for 2nd words
- Once we have sets of possible words that can be made from those letters, can we put them together in a board?

**Goals**
- Is this set solvable?
- How many ways are there to solve this set?
- What are those ways?

**Extensions**
- Letter classifier so that we can input image of dice and then get the outputs
- Board visualizer of possible outputs
- Find a set of dice that maximizes probability that a roll is solvable


**Taking Pictures of Dice**
-TODO: Take some examples of good images
For best model performance in selecting the correct dice in the pictures you take:
- Roll the dice on some printer paper or otherwise solid-colored surface
- Bring the dice together so they fit in a roughly printer-paper sized area, but leave them about evenly spaced within that area
- Take a picture that includes all dice in as clear detail as possible, and don't use the flash

**Sources**
- [Appel and Jacobson Scrabble Paper](https://www.cs.cmu.edu/afs/cs/academic/class/15451-s06/www/lectures/scrabble.pdf)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Chars74k Dataset](https://teodecampos.github.io/chars74k/)


**Troubleshooting**
- When trying to download EMNIST data, if getting 'zipfile.BadZipFile: File is not a zip file', download the emnist dataset from the website, rename gzip.zip to emnist.zip, and move it to your home directory at ~/.cache/emnist

**TODO**
- find_dice.py: Convert everything possible that uses skimage to use opencv (i.e. transform, but probably not measure)

**Ideas For Improvement**
- Somehow rotate our letters before predicting on them
- Both EMNIST and Chars are vertically oriented, so we could augment for rotation pretty easily