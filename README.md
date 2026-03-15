
Q-less solver by Indy and Emerson

**Running It**
All runnable scripts listed below have example commands for running at the top of the file.
- `pipeline.py`:
- `modeling_torch.py`: For training and testing models using PyTorch
- `modeling_custom.py`: For training and testing models using our custom CNN architecture

**TODO**
- Make UI that shows the dice image and the predictions for each dice right next to each die so that user can verify themselves if all of the letters are right?

**Ideas For Improvement**
- Synthetic dataset generation using Q-Less font

**Extensions**
- How many ways are there to solve a set of dice?
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
- [How do we know if we made decent letter predictions? What are all of the possible 12 letter words? (Bipartite Matching)](https://www.geeksforgeeks.org/dsa/maximum-bipartite-matching/)
- [PyTorch CNN Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Backpropagation for Convolutional Layers](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)

**Troubleshooting**
- When trying to download EMNIST data, if getting 'zipfile.BadZipFile: File is not a zip file', download the emnist dataset from the website, rename gzip.zip to emnist.zip, and move it to your home directory at ~/.cache/emnist

**All the 12 letter words!**
astringently
backdropping
backflipping
backlighting
backslapping
backtracking
barnstorming
battlefronts
benchmarking
besprinkling
bewitchingly
birdwatching
blacklisters
blacklisting
blacktopping
blisteringly
blockbusting
blowtorching
blusteringly
bodychecking
bowstringing
bryophyllums
bullfighters
bushwhacking
butterflying
candlelights
checkmarking
chloroplasts
chromoplasts
clatteringly
comfortingly
comptrollers
copyrighting
cryptanalyst
cryptarithms
cryptologist
cryptorchism
crystalizing
cyclostyling
daylightings
depressingly
despondently
disbranching
discrepantly
dislodgments
disportments
distraughtly
farsightedly
filmsettings
flowcharting
frostbitings
gangsterdoms
glassblowing
grandmothers
groundswells
hairstylings
hamstringing
handstamping
highbrowisms
hornswoggled
hydrologists
hydrospheres
hydrospheric
hydrotropism
hyperbolists
hypercomplex
hypercritics
hyperplastic
lamplighters
landholdings
lightweights
locksmithing
lymphoblasts
matchmarking
mirthfulness
nightclubbed
nightclubber
nightwalkers
paddywacking
pennyweights
pennywhistle
persistently
phragmoplast
phycologists
phytosterols
pitchblendes
plasterworks
platyrrhines
playwritings
podophyllins
podophyllums
polycentrism
polychroming
polycrystals
postmidnight
printmakings
prognathisms
propertyless
prosobranchs
prostacyclin
prothrombins
protomartyrs
psychologist
psychrometer
psychrometry
pterodactyls
rightfulness
scatteringly
schnorkeling
scintigraphy
scrimshawing
shatteringly
shellackings
skyrocketing
smallholding
spaceflights
spectrograms
spellbinding
sprightliest
springboards
staphylinids
stepbrothers
stepchildren
stickhandled
stickhandler
stockbroking
stockholders
storytelling
straightbred
stranglehold
stratigraphy
strawflowers
streptolysin
stringhalted
stringybarks
swelteringly
switchbacked
swordplayers
thallophytes
thriftlessly
thunderingly
thyrotrophin
thyrotropins
tracklayings
transcribing
trellisworks
trestleworks
trichlorfons
trichlorphon
triphthongal
triumphantly
trophoblasts
twelvemonths
typewritings
unsettlingly
walkingstick
watchmakings
weightlessly
xanthophylls

**Some stats**
Number of unique roll outcomes: 15800439          
Total possible combinations calculated: 2176782336