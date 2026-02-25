
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
- Make full pipeline from image to letter predictions
- Making starting word be as long as possible for recursive efficiency

**Ideas For Improvement**
- Somehow rotate our letters before predicting on them
- Both EMNIST and Chars are vertically oriented, so we could augment for rotation pretty easily

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