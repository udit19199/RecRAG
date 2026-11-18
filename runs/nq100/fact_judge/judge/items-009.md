Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=735422d3493eaed4 ===
Gold sentences and facts:
Sentence 0: For thousands of years , seawater pearls were retrieved by divers in the Indian Ocean in areas such as the Persian Gulf , the Red Sea and the Gulf of Mannar . Evidence also suggest a prehistoric origin to pearl diving in these regions . Starting in the Han Dynasty ( 206 BC -- 220 AD ) , the Chinese hunted extensively for seawater pearls in the South China Sea . In the 14th - century Arabian Sea , the traveller Ibn Battuta provided the earliest known description of pearl diving by means of attaching a cord to the diver 's waist .
  Fact 0.0: For thousands of years, seawater pearls were retrieved by divers in the Indian Ocean.
  Fact 0.1: Divers retrieved seawater pearls in areas such as the Persian Gulf.
  Fact 0.2: Divers retrieved seawater pearls in areas such as the Red Sea.
  Fact 0.3: Divers retrieved seawater pearls in areas such as the Gulf of Mannar.
  Fact 0.4: Evidence also suggests a prehistoric origin to pearl diving in these regions.
  Fact 0.5: Starting in the Han Dynasty (206 BC to 220 AD), the Chinese hunted extensively for seawater pearls in the South China Sea.
  Fact 0.6: In the 14th-century Arabian Sea, the traveller Ibn Battuta provided the earliest known description of pearl diving.
  Fact 0.7: Ibn Battuta's description of pearl diving involved attaching a cord to the diver's waist.

Retrieved items:
<item 1>
Pearl
1 British Isles 7.2 Pearl farming 8 Timeline of pearl production 9 Freshwater pearl farming 10 Momme weight 11 In jewelry 11.1 Gallery 11.2 Shapes 11.3 Lengths
of pearl necklaces 11.4 Colors 12 Religious references 12.1 Hindu scriptures 12.2 Hebrew scriptures 12.3 New Testament scriptures 12.4 Islamic scriptures 12.5
Additional references 13 See also 14 References 15 External links Etymology ( edit ) The English word pearl comes from the French perle , originally from the
Latin perna meaning leg , after the ham - or mutton leg - shaped bivalve . Definition ( edit ) A black pearl and a shell of the black - lipped pearl oyster .
The iridescent colors originate from nacre layers . All shelled mollusks can , by natural processes , produce some kind of `` pearl '' when an irritating
microscopic object becomes trapped within its mantle folds , but the great majority of these `` pearls '' are not valued as gemstones .
</item 1>

<item 2>
Pearl
August 13 , 2013 . Archived from the original on February 21 , 2014 . Jump up ^ Kunz , George F. ; Stevenson , Charles ( 1908 ) . The book of the pearl . New
York : The Century Co. p. 412 . Archived from the original on May 14 , 2016 . Jump up ^ Kunz , George F. ; Stevenson , Charles ( 1908 ) . The book of the pearl
. New York : The Century Co. p. 350 . Archived from the original on May 14 , 2016 . External links ( edit ) Wikimedia Commons has media related to : Pearl (
category ) Wikisource has the text of the 1921 Collier 's Encyclopedia article Pearl . The History of Pearls . PBS Pearl History Special . Recover the major
pearl produce country by aquaculture in UAE ( Japanese page with English narration ) Jewellery Forms Anklet Barrette Belt buckle Belly chain Bindi Bolo tie
Bracelet Brooch Chatelaine Collar pin Crown Cufflink Earring Ferronnière Lapel pin Necklace Pectoral Pendant Ring Tiara Tie
</item 2>

<item 3>
Pearl
It is a naturally occurring , non-nacreous , calcareous concretion ( pearl ) from a giant clam . Because it did not grow in a pearl oyster it is not pearly ;
instead the surface is glossy like porcelain . Other pearls from giant clams are known to exist , but this is a particularly large one weighing 14 lb ( 6.4 kg )
. The largest known pearl ( also from a giant clam ) was found in the Philippines as well by a fisherman from Puerto Princesa , Palawan Island . The enormous
pearl is 30 cm wide ( 1 ft ) , 67 cm long ( 2.2 ft ) and weighs 75 lb ( 34 kg ) . History ( edit ) Pearl hunting ( edit ) Main article : Pearl hunting A 14th -
century piece of clothing used by Kuwaiti divers searching for pearls in the Persian Gulf The ancient chronicle Mahavamsa mentions of the thriving pearl
industry in the port of Oruwella in the Gulf of Mannar in Sri Lanka .
</item 3>

<item 4>
Pearl
Australia also has one of the world 's last remaining fleets of pearl diving ships . Australian pearl divers dive for south sea pearl oysters to be used in the
cultured south sea pearl industry . The catch of pearl oysters is similar to the numbers of oysters taken during the natural pearl days . Hence significant
numbers of natural pearls are still found in the Australian Indian Ocean waters from wild oysters . X-ray examination is required to positively verify natural
pearls found today . Types of cultured pearls ( edit ) A blister pearl , a half - sphere , formed flush against the shell of the pearl oyster . Keshi pearls ,
although they often occur by chance , are not considered natural . They are a byproduct of the culturing process , and hence do not happen without human
intervention . They are quite small , typically only a few millimeters . Keshi pearls are produced by many different types of marine mollusks and freshwater
mussels in China . Keshi pearls are actually a mistake in the cultured pearl seeding process . In seeding the cultured pearl , a piece of mantle muscle from a
sacrificed oyster is placed with a bead of mother of pearl within the oyster .
</item 4>

<item 5>
Pearl
Prized for their exquisitely beautiful ' orient ' or lustre , South Sea pearls are now farmed in various parts of the world where the Pinctada maxima oysters
can be found , with the finest South Sea pearls being produced by Paspaley along the remote coastline of North - Western Australia . White and silver colored
South Sea pearls tend to come from the Broome area of Australia , while golden colored ones are more prevalent in the Philippines and Indonesia . A farm in the
Gulf of California , Mexico , is culturing pearls from the black lipped Pinctada mazatlanica oysters and the rainbow lipped Pteria sterna oysters . Also called
Concha Nácar , the pearls from these rainbow lipped oysters fluoresce red under ultraviolet light . From other species ( edit ) A shell of the Indian volute ,
Melo melo , surrounded by a number of pearls from this species Conch pearl pendant Biologically speaking , under the right set of circumstances , almost any
shelled mollusk can produce some kind of pearl . However , most of these molluskan pearls have no luster or iridescence .
</item 5>

<item 6>
Pearl
Jump up ^ Senaveratna , John M ( 1930 ) . The story of the Sinhalese : From the most ancient times up to the end of `` The Mahavasna '' , or great dynasty ;
Vijaya to Maha Sena ( B.C. 543 to A.D. 302 ) . W.M.A. Wahid . pp. 16 -- 9 , 46,130,144 -- 8,161 -- 3,213 . ASIN B000874SS0 . ISBN 9788120612716 . Archived from
the original on December 20 , 2017 . Jump up ^ Kunz , George F. ; Stevenson , Charles ( 1908 ) . The book of the pearl . New York : The Century Co. p. 2 .
Archived from the original on May 14 , 2016 . Jump up ^ R. Raghu Prasad and P.V. Ramachandran Nair ( 1973 ) . `` India and the Indian Ocean Fisheries '' ( PDF )
. Journal of the Marine Biological Association of India .
</item 6>

<item 7>
Pearl
It also records that eight varieties of pearls accompanied Prince Vijaya 's embassy to the Pandyan king as well as king Devanampiya Tissa 's embassy to Emperor
Ashoka . Pliny the Elder ( 23 -- 79AD ) praised the pearl fishery of the Gulf as most productive in the world . For thousands of years , seawater pearls were
retrieved by divers in the Indian Ocean in areas such as the Persian Gulf , the Red Sea and the Gulf of Mannar . Evidence also suggest a prehistoric origin to
pearl diving in these regions . Starting in the Han Dynasty ( 206 BC -- 220 AD ) , the Chinese hunted extensively for seawater pearls in the South China Sea .
In the 14th - century Arabian Sea , the traveller Ibn Battuta provided the earliest known description of pearl diving by means of attaching a cord to the diver
's waist . Catching of pearls , Bern Physiologus ( 9th century ) When Spanish conquistadors arrived in the Western Hemisphere , they discovered that around the
islands of Cubagua and Margarita , some 200 km north of the Venezuelan coast , was an extensive pearl bed ( a bed of pearl oysters ) .
</item 7>

<item 8>
Pearl
Pearls of abalone , or pāua , are mabe pearls , or blister pearls , unique to New Zealand waters and are commonly referred to as ' blue pearls ' . They are
admired for their incredible luster and naturally bright vibrant colors that are often compared to opal . Another example is the conch pearl ( sometimes
referred to simply as the ' pink pearl ' ) , which is found very rarely growing between the mantle and the shell of the queen conch or pink conch , Strombus
gigas , a large sea snail or marine gastropod from the Caribbean Sea . These pearls , which are often pink in color , are a by - product of the conch fishing
industry , and the best of them display a shimmering optical effect related to chatoyance known as ' flame structure ' . Somewhat similar gastropod pearls ,
this time more orange in hue , are ( again very rarely ) found in the horse conch Pleuroploca gigantea . The second largest pearl known was found in the
Philippines in 1934 and is known as the Pearl of Lao Tzu .
</item 8>

<item 9>
Pearl
Journal of the Marine Biological Association of India . 15 : 1 -- 19 . Archived ( PDF ) from the original on July 18 , 2011 . Jump up ^ Arnold Wright ( 1999 ) .
Twentieth century impressions of Ceylon : its history , people , commerce , industries , and resources . p. 227 . ISBN 81 - 206 - 1335 - X . Archived from the
original on January 18 , 2017 . Jump up ^ James Hornell ( 2009 ) . The Indian Pearl Fisheries of the Gulf of Manar and Palk Bay . BiblioBazaar . p. 6 . ISBN 1 -
110 - 87096 - 5 . Archived from the original on January 18 , 2017 . Jump up ^ De Silva , K.M. ( 1995 ) . Volume 2 of History of Ceylon , History of Ceylon :
History of Sri Lanka . Peradeniya : Ceylon University Press . p. 56 . ISBN 955 - 589 - 004 - 8 .
</item 9>

<item 10>
Pearl
While white , and more recently black , saltwater pearls are by far the most popular , other color tints can be found on pearls from the oceans . Pink , blue ,
champagne , green , black and even purple saltwater pearls can be encountered , but to collect enough of these rare colors to form a complete string of the same
size and same shade can take years . Religious references ( edit ) This section needs additional citations for verification . Please help improve this article
by adding citations to reliable sources . Unsourced material may be challenged and removed . ( March 2018 ) ( Learn how and when to remove this template message
) Hindu scriptures ( edit ) The Hindu tradition describes the sacred Nine Pearls which were first documented in the Garuda Purana , one of the books of the
Hindu mythology . Ayurveda contains references to pearl powder as a stimulant of digestion and to treat mental ailments . According to Marco Polo , the kings of
Malabar wore a necklace of 104 rubies and pearls which was given from one generation of kings to the next . The reason was that every king had to say 104
prayers every morning and every evening .
</item 10>
=== END ITEM id=735422d3493eaed4 ===

=== ITEM id=53b74941d353abf0 ===
Gold sentences and facts:
Sentence 0: Country Name of animal Scientific name Pictures Ref . Algeria Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer ( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian manatee ( national aquatic animal ) Trichechus manatus Croatia Pine marten Martes martes Cuba Cuban trogon Priotelus temnurus Cyprus Cypriot mouflon Ovis orientalis Czech Republic Double - tailed lion Mythical Denmark Mute swan ( national bird ) Cygnus olor Small tortoiseshell ( national butterfly ) Aglais urticae Egypt Steppe eagle Aquila nipalensis Estonia Barn swallow ( national bird ) Hirundo rustica Eritrea Arabian camel Camelus dromedarius Ethiopia Lion Panthera leo Faroe Islands Eurasian oystercatcher ( national bird ) Haematopus ostralgus Finland Brown bear ( national animal ) Ursus arctos Whooper swan ( national bird ) Cygnus cygnus Holly blue ( national butterfly ) Celastrina argiolus European perch ( national fish ) Perca fluviatilis Seven - spot ladybird ( national insect ) Coccinella septempunctata France Gallic rooster Gallus gallus domesticus Germany Unspecified Eagle Members of Accipitridae Greece Phoenix Mythical Dolphins Delphinus Guatemala Quetzal Pharomachrus mocinno Haiti Hispaniolan Trogan Priotelus roseigaster Honduras White - tailed deer Odocoileus virginianus Hungary Turul Mythical ( most probably Falco cherrug ) Iceland Gyrfalcon Falco rusticolus India Bengal tiger ( National predator ) Panthera tigris tigris Indian peafowl ( National bird ) Pavo cristatus King cobra ( National reptile ) Ophiophagus hannah Ganges river dolphin ( National aquatic animal ) Platanista gangetica gangetica Indian elephant ( National heritage animal ) Elephas maximus indicus Indonesia Komodo dragon ( national animal ) Varanus komodoensis Garuda ( historic national animal ) Mythical Iran Persian lion Panthera leo persica Iraq Chukar partridge Alectoris chukar Goat ( KRG ) Capra aegagrus Israel Hoopoe ( national bird ) Upupa epops Jamaica Red - billed streamertail Trochilus polytmus Japan Green pheasant ( national bird ) Phasianus versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal ) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod ) Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle ( national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison
  Fact 0.0: The table lists the fennec fox (Vulpes zerda) for Algeria.
  Fact 0.1: The table lists the red-crested turaco (Tauraco erythrolophus) as Angola's national bird.
  Fact 0.2: The table lists the Zenaida dove (Zenaida aurita) for Anguilla.
  Fact 0.3: Antigua and Barbuda's national animal is the fallow deer (Dama dama).
  Fact 0.4: Antigua and Barbuda's national bird is the frigate (Fregata magnificens).
  Fact 0.5: Antigua and Barbuda's national sea creature is the hawksbill turtle (Eretmochelys imbricata).
  Fact 0.6: The table lists the rufous hornero (Furnarius rufus) for Argentina.
  Fact 0.7: Australia's national animal is the red kangaroo (Macropus rufus).
  Fact 0.8: Australia's national bird is the emu (Dromaius novaehollandiae).
  Fact 0.9: The table lists the black eagle (Ictinaetus malaiensis) for Austria.
  Fact 0.10: The table lists the Karabakh horse (Equus ferus caballus) for Azerbaijan.
  Fact 0.11: Bangladesh's national animal is the Royal Bengal tiger (Panthera tigris tigris).
  Fact 0.12: Bangladesh's national bird is the magpie robin (Copsychus saularis).
  Fact 0.13: Bangladesh's national fish is the ilish (Tenualosa ilisha).
  Fact 0.14: The table lists the European bison (Bison bonasus) for Belarus.
  Fact 0.15: The table lists the lion (heraldic Leo Belgicus, Panthera leo) for Belgium.
  Fact 0.16: Belize's national animal is Baird's tapir (Tapirus bairdii).
  Fact 0.17: Belize's national bird is the keel-billed toucan (Ramphastos sulfuratus).
  Fact 0.18: The table lists the Druk Mythical Takin (Budorcas taxicolor) for Bhutan.
  Fact 0.19: The table lists the rufous-bellied thrush (Turdus rufiventris) for Brazil.
  Fact 0.20: The table lists the kouprey (Bos sauveli) for Cambodia.
  Fact 0.21: Canada's sovereignty animal symbol is the North American beaver (Castor canadensis).
  Fact 0.22: Canada's national horse is the Canadian horse (Equus ferus caballus).
  Fact 0.23: China's national animal is the giant panda (Ailuropoda melanoleuca).
  Fact 0.24: China's national animal is the Chinese dragon, with scientific name listed as Mythical.
  Fact 0.25: China's national bird is the red-crowned crane (Grus japonensis).
  Fact 0.26: The table lists the okapi (Okapia johnstoni) for the Democratic Republic of the Congo.
  Fact 0.27: The table lists the Andean condor (Vultur gryphus) for Colombia.
  Fact 0.28: Costa Rica's national bird is the yigüirro (Turdus grayi).
  Fact 0.29: Costa Rica's national animal is the white-tailed deer (Odocoileus virginianus).
  Fact 0.30: Costa Rica's national aquatic animal is the West Indian manatee (Trichechus manatus).
  Fact 0.31: The table lists the pine marten (Martes martes) for Croatia.
  Fact 0.32: The table lists the Cuban trogon (Priotelus temnurus) for Cuba.
  Fact 0.33: The table lists the Cypriot mouflon (Ovis orientalis) for Cyprus.
  Fact 0.34: The table lists the double-tailed lion, with scientific name listed as Mythical, for the Czech Republic.
  Fact 0.35: Denmark's national bird is the mute swan (Cygnus olor).
  Fact 0.36: Denmark's national butterfly is the small tortoiseshell (Aglais urticae).
  Fact 0.37: The table lists the steppe eagle (Aquila nipalensis) for Egypt.
  Fact 0.38: Estonia's national bird is the barn swallow (Hirundo rustica).
  Fact 0.39: The table lists the Arabian camel (Camelus dromedarius) for Eritrea.
  Fact 0.40: The table lists the lion (Panthera leo) for Ethiopia.
  Fact 0.41: The Faroe Islands' national bird is the Eurasian oystercatcher (Haematopus ostralgus).
  Fact 0.42: Finland's national animal is the brown bear (Ursus arctos).
  Fact 0.43: Finland's national bird is the whooper swan (Cygnus cygnus).
  Fact 0.44: Finland's national butterfly is the holly blue (Celastrina argiolus).
  Fact 0.45: Finland's national fish is the European perch (Perca fluviatilis).
  Fact 0.46: Finland's national insect is the seven-spot ladybird (Coccinella septempunctata).
  Fact 0.47: The table lists the Gallic rooster (Gallus gallus domesticus) for France.
  Fact 0.48: The table lists an unspecified eagle (members of Accipitridae) for Germany.
  Fact 0.49: The table lists the phoenix, with scientific name listed as Mythical, for Greece.
  Fact 0.50: The table lists dolphins (Delphinus) for Greece.
  Fact 0.51: The table lists the quetzal (Pharomachrus mocinno) for Guatemala.
  Fact 0.52: The table lists the Hispaniolan trogan (Priotelus roseigaster) for Haiti.
  Fact 0.53: The table lists the white-tailed deer (Odocoileus virginianus) for Honduras.
  Fact 0.54: The table lists the turul, with scientific name listed as Mythical (most probably Falco cherrug), for Hungary.
  Fact 0.55: The table lists the gyrfalcon (Falco rusticolus) for Iceland.
  Fact 0.56: India's national predator is the Bengal tiger (Panthera tigris tigris).
  Fact 0.57: India's national bird is the Indian peafowl (Pavo cristatus).
  Fact 0.58: India's national reptile is the king cobra (Ophiophagus hannah).
  Fact 0.59: India's national aquatic animal is the Ganges river dolphin (Platanista gangetica gangetica).
  Fact 0.60: India's national heritage animal is the Indian elephant (Elephas maximus indicus).
  Fact 0.61: Indonesia's national animal is the Komodo dragon (Varanus komodoensis).
  Fact 0.62: Indonesia's historic national animal is the garuda, with scientific name listed as Mythical.
  Fact 0.63: The table lists the Persian lion (Panthera leo persica) for Iran.
  Fact 0.64: The table lists the chukar partridge (Alectoris chukar) for Iraq.
  Fact 0.65: The table lists the goat (Capra aegagrus) for the Iraqi Kurdistan Region (KRG).
  Fact 0.66: Israel's national bird is the hoopoe (Upupa epops).
  Fact 0.67: The table lists the red-billed streamertail (Trochilus polytmus) for Jamaica.
  Fact 0.68: Japan's national bird is the green pheasant (Phasianus versicolor).
  Fact 0.69: Japan's national fish is the carp (Cyprinus carpio).
  Fact 0.70: Latvia's national bird is the white wagtail (Motacilla alba).
  Fact 0.71: Latvia's national insect is the two-spotted ladybird (Adalia bipunctata).
  Fact 0.72: The table lists the white stork (Ciconia ciconia) for Lithuania.
  Fact 0.73: The table lists the lion (Panthera leo), in Macedonian heraldry, for Macedonia.
  Fact 0.74: The table lists the zebu (Bos primigenius indicus), in Malagasy heraldry, for Madagascar.
  Fact 0.75: Madagascar's national animal is the ring-tailed lemur (Lemur catta).
  Fact 0.76: Malaysia's national animal is the Malayan tiger (Panthera tigris jacksoni).
  Fact 0.77: Malta's national animal is the Pharaoh Hound (Canis lupus familiaris (1)).
  Fact 0.78: Moldova's national animal is the aurochs (Bos primigenius).
  Fact 0.79: The table lists the dodo (Raphus cucullatus) for Mauritius.
  Fact 0.80: Mexico's national animal is the golden eagle (Aquila chrysaetos).
  Fact 0.81: Mexico's national dog is the xoloitzcuintli (Canis lupus familiaris).
  Fact 0.82: Mexico's national mammal is the jaguar (Panthera onca).
  Fact 0.83: Mexico's national arthropod is the grasshopper (Sphenarium purpurascens).
  Fact 0.84: Mexico's national marine mammal is the vaquita (Phocoena sinus).
  Fact 0.85: Nepal's national animal is the cow (Bos taurus indicus).
  Fact 0.86: Nepal's national bird is the danphe (Lophophorus impejanus).
  Fact 0.87: The Netherlands' heraldic symbol is the lion (Panthera leo).
  Fact 0.88: The Netherlands' national bird is the black-tailed godwit (Limosa limosa).
  Fact 0.89: The table lists the kiwi (Apteryx sp.) for New Zealand.
  Fact 0.90: The table lists the chollima, with scientific name listed as Mythical, for North Korea.
  Fact 0.91: Norway's royal national animal is the lion (Panthera leo).
  Fact 0.92: Norway's national bird is the white-throated dipper (Cinclus cinclus).
  Fact 0.93: Norway's national horse is the Fjord horse (Equus ferus caballus).
  Fact 0.94: Oman's national animal is the Arabian oryx (Oryx leucoryx).
  Fact 0.95: Pakistan's national animal is the markhor (Capra falconeri).
  Fact 0.96: Pakistan's national aquatic mammal is the Indus river dolphin (Platanista indicus minor minor).
  Fact 0.97: Pakistan's national bird is the chukar (Alectoris chukar).
  Fact 0.98: Pakistan's national reptile is the Indus crocodile (Crocodylus palustris).
  Fact 0.99: Pakistan's heritage bird is the shaheen falcon (Falco peregrinus peregrinator).
  Fact 0.100: Pakistan's national predator is the snow leopard (Panthera uncia).
  Fact 0.101: The table lists the harpy eagle (Harpia harpyja) for Panama.
  Fact 0.102: Papua New Guinea's national marine mammal is the dugong (Dugong dugon).
  Fact 0.103: Peru's national animal is the vicuña (Vicugna vicugna).
  Fact 0.104: The Philippines' national animal is the carabao (Bubalus bubalis).
  Fact 0.105: The Philippines' national bird is the Philippine eagle (Pithecophaga jefferyi).
  Fact 0.106: Poland's national bird is the white-tailed eagle (Haliaeetus albicilla).
  Fact 0.107: Poland's national bird is also the white stork (Ciconia ciconia).
  Fact 0.108: Poland's national animal is the European bison (Bison bonasus).
  Fact 0.109: The table lists the Arabian oryx (Oryx leucoryx) for Qatar.
  Fact 0.110: The table lists the lynx (Lynx lynx) for Romania.
  Fact 0.111: The table lists the Eurasian brown bear (Ursus arctos arctos) for Russia.
  Fact 0.112: The table lists the leopard (Panthera pardus) for Rwanda.
  Fact 0.113: The table lists the Saint Vincent amazon (Amazona guildingii) for Saint Vincent and the Grenadines.
  Fact 0.114: The table lists the wolf (Canis lupus lupus) for Serbia.
  Fact 0.115: The table lists the white eagle, with scientific name listed as Mythical, for Serbia.
  Fact 0.116: The table lists the lynx (Lynx lynx) for Serbia.
  Fact 0.117: Somalia's national animal is the leopard (Panthera pardus).
  Fact 0.118: South Africa's national animal is the springbok (Antidorcas marsupialis).
  Fact 0.119: South Africa's national bird is the blue crane (Anthropoides paradiseus).
  Fact 0.120: South Africa's national fish is the galjoen (Dichistius capensis).
  Fact 0.121: The table lists the Siberian tiger (Panthera tigris altaica) for South Korea.
  Fact 0.122: South Korea's national bird is the Korean magpie (Pica (pica) sericea).
  Fact 0.123: Sri Lanka's national bird is the Sri Lankan junglefowl (Gallus lafayettii).
  Fact 0.124: The table lists the giraffe (Giraffa camelopardalis) for Tanzania.
  Fact 0.125: The table lists the elephant (Elephas maximus) for Thailand.
  Fact 0.126: The United Arab Emirates' national animal is the Arabian oryx (Oryx leucoryx).
  Fact 0.127: The United Arab Emirates' national bird is the peregrine falcon (Falco peregrinus).
  Fact 0.128: The table lists the lion (Panthera leo) for the United Kingdom, England, Scotland, and Wales.
  Fact 0.129: The table lists the British bulldog (Canis lupus familiaris) for Britain.
  Fact 0.130: The table lists the red kite (Milvus milvus) for Wales.
  Fact 0.131: The table lists the unicorn, with scientific name listed as Mythical, for Scotland.
  Fact 0.132: The table lists the Y Ddraig Goch (Welsh Dragon), with scientific name listed as Mythical, for Wales.
  Fact 0.133: The United States' national bird is the bald eagle (Haliaeetus leucocephalus).
  Fact 0.134: The United States' national mammal is the American bison (Bison bison).

Retrieved items:
<item 1>
List of national animals
Greece Phoenix Mythical Dolphins Delphinus Guatemala Quetzal Pharomachrus mocinno Haiti Hispaniolan Trogan Priotelus roseigaster Honduras White - tailed deer
Odocoileus virginianus Hungary Turul Mythical ( most probably Falco cherrug ) Iceland Gyrfalcon Falco rusticolus India Bengal tiger ( National predator )
Panthera tigris tigris Indian peafowl ( National bird ) Pavo cristatus King cobra ( National reptile ) Ophiophagus hannah Ganges river dolphin ( National
aquatic animal ) Platanista gangetica gangetica Indian elephant ( National heritage animal ) Elephas maximus indicus Indonesia Komodo dragon ( national animal )
Varanus komodoensis Garuda ( historic national animal ) Mythical Iran Persian lion Panthera leo persica Iraq Chukar partridge Alectoris chukar Goat ( KRG )
Capra aegagrus Israel Hoopoe ( national bird ) Upupa epops Jamaica Red - billed streamertail Trochilus polytmus Japan Green pheasant ( national bird ) Phasianus
versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail (
</item 1>

<item 2>
List of national animals
Canadian Heritage . 2016 . Retrieved 1 December 2016 . Jump up ^ `` National Symbols of China '' . 123independenceday.com . Jump up ^ `` China National Symbols
'' . worldatlast.com. 2015 . Retrieved 2017 - 03 - 28 . Jump up ^ `` China Considers Red - crowned Crane for National Bird '' . www.china.org.cn . Jump up ^ ``
El cóndor , ave nacional - banrepcultural.org '' . Jump up ^ `` 16 de diciembre : Declaración del yigüirro como ave nacional de Costa Rica '' . Jump up ^ `` El
Venado Cola Blanca , Símbolo Nacional de la Fauna Silvestre '' . Jump up ^ `` Manatee Now Costa Rica 's New National Symbol '' . QCOSTARICA . Archived from the
original on 14 September 2014 . Jump up ^ Bousfield , Jonathan ( 2003 ) . Croatia . Rough Guides . p. 72 . ISBN 1843530848 .
</item 2>

<item 3>
List of national animals
Magpie robin ( national bird ) Copsychus saularis Ilish ( national fish ) Tenualosa ilisha Belarus European bison Bison bonasus Belgium Lion ( heraldic Leo
Belgicus ) Panthera leo Belize Baird 's tapir ( national animal ) Tapirus bairdii Keel - billed toucan ( national bird ) Ramphastos sulfuratus Bhutan Druk
Mythical Takin Budorcas taxicolor Brazil Rufous - bellied thrush Turdus rufiventris Cambodia Kouprey Bos sauveli Canada North American beaver ( sovereignty
animal symbol ) Castor canadensis Canadian horse ( national horse ) Equus ferus caballus China Giant panda ( national animal ) Ailuropoda melanoleuca Chinese
dragon ( national animal ) Mythical Red - crowned crane ( national bird ) Grus japonensis Democratic Republic of the Congo Okapi Okapia johnstoni Colombia
Andean condor Vultur gryphus Costa Rica Yigüirro ( national bird ) Turdus grayi White - tailed deer ( national animal ) Odocoileus virginianus West Indian
manatee ( national aquatic animal ) Trichechus manatus Croatia Pine
</item 3>

<item 4>
List of national animals
Vicugna vicugna Philippines Carabao ( national animal ) Bubalus bubalis Philippine eagle ( national bird ) Pithecophaga jefferyi Poland White - tailed eagle (
national bird ) Haliaeetus albicilla White stork ( national bird ) Ciconia ciconia European bison ( national animal ) Bison bonasus Qatar Arabian oryx Oryx
leucoryx Romania Lynx Lynx lynx Russia Eurasian brown bear Ursus arctos arctos Rwanda Leopard Panthera pardus Saint Vincent and the Grenadines Saint Vincent
amazon Amazona guildingii Serbia Wolf Canis lupus lupus White eagle Mythical Lynx Lynx lynx Somalia Leopard ( national animal ) Panthera pardus South Africa
Springbok ( national animal ) Antidorcas marsupialis Blue crane ( national bird ) Anthropoides paradiseus Galjoen ( national fish ) Dichistius capensis South
Korea Siberian tiger Panthera tigris altaica Korean magpie ( national bird ) Pica ( pica ) sericea Sri Lanka Sri Lankan junglefowl ( national bird ) Gallus
lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant
</item 4>

<item 5>
List of national animals
Danphe ( national bird ) Lophophorus impejanus Netherlands Lion ( heraldic symbol ) Panthera leo Black - tailed godwit ( national bird ) Limosa limosa New
Zealand Kiwi Apteryx sp . North Korea Chollima Mythical Norway Lion ( royal national animal ) Panthera leo White - throated dipper ( national bird ) Cinclus
cinclus Fjord horse ( national horse ) Equus ferus caballus Oman Arabian oryx ( national animal ) Oryx leucoryx Pakistan Markhor ( National animal ) Capra
falconeri Indus river dolphin ( National aquatic mammal ) Platanista indicus minor minor Chukar ( National bird ) Alectoris chukar Indus crocodile ( National
reptile ) Crocodylus palustris Shaheen falcon ( Heritage bird ) Falco peregrinus peregrinator Snow leopard ( National predator ) Panthera uncia Panama Harpy
eagle Harpia harpyja Papua New Guinea Dugong ( national marine mammal ) Dugong dugon Peru Vicuña ( national animal ) Vicugna vicugna Philippines Carabao (
national animal ) Bubalus bubalis Philippine
</item 5>

<item 6>
List of national animals
versicolor Carp ( national fish ) Cyprinus carpio Latvia White wagtail ( national bird ) Motacilla alba Two - spotted ladybird ( national insect ) Adalia
bipunctata Lithuania White stork Ciconia ciconia Macedonia Lion ( in Macedonian heraldry ) Panthera leo Madagascar Zebu ( in Malagasy heraldry ) Bos primigenius
indicus Ring - tailed lemur ( national animal ) Lemur catta Malaysia Malayan tiger ( national animal ) Panthera tigris jacksoni Malta Pharaoh Hound ( national
animal ) Canis lupus familiaris ( 1 ) Moldova Aurochs ( national animal ) Bos primigenius Mauritius Dodo Raphus cucullatus Mexico Golden eagle ( national animal
) Aquila chrysaetos Xoloitzcuintli ( national dog ) Canis lupus familiaris Jaguar ( national mammal ) Panthera onca Grasshopper ( national arthropod )
Sphenarium purpurascens Vaquita ( national marine mammal ) Phocoena sinus Nepal Cow ( national animal ) Bos taurus indicus Danphe ( national bird ) Lophophorus
impejanus Netherlands Lion ( heraldic symbol
</item 6>

<item 7>
List of national animals
bird ) Gallus lafayettii Tanzania Giraffe Giraffa camelopardalis Thailand Elephant Elephas maximus United Arab Emirates Arabian oryx ( national animal ) Oryx
leucoryx Peregrine falcon ( national bird ) Falco peregrinus United Kingdom Lion ( United Kingdom , England , Scotland , Wales ) Panthera leo Bulldog ( Britain
as `` British Bulldog '' ) Canis lupus familiaris Red kite ( Wales ) Milvus milvus Unicorn ( Scotland ) Mythical Y Ddraig Goch ( Welsh Dragon Wales ) Mythical
United States Bald eagle ( national bird ) Haliaeetus leucocephalus American bison ( national mammal ) Bison bison See also Animals portal Lists portal State
animal List of national birds Animals as heraldic charges Floral emblem National personification References Jump up ^ `` The Fennec Fox : Algeria 's National
Animal '' . algeria.com . Retrieved 2013 - 03 - 21 . Jump up ^ `` THE NATIONAL BIRD OF ANGOLA '' . BIRDS ANGOLA . Retrieved 12 April 2016 .
</item 7>

<item 8>
List of national animals
List of National Animals - wikipedia List of National Animals Jump to : navigation , search This article is incomplete . Please help to improve it , or discuss
the issue on the talk page . ( June 2017 ) This is a list of national animals . National animals Country Name of animal Scientific name Pictures Ref . Algeria
Fennec fox Vulpes zerda Angola Red - crested turaco ( national bird ) Tauraco erythrolophus Anguilla Zenaida dove Zenaida aurita Antigua and Barbuda Fallow deer
( national animal ) Dama dama Frigate ( national bird ) Fregata magnificens Hawksbill turtle ( national sea creature ) Eretmochelys imbricata Argentina Rufous
hornero Furnarius rufus Australia Red kangaroo ( national animal ) Macropus rufus Emu ( national bird ) Dromaius novaehollandiae Austria Black eagle Ictinaetus
malaiensis Azerbaijan Karabakh horse Equus ferus caballus Bangladesh Royal Bengal tiger ( national animal ) Panthera tigris tigris Magpie robin ( national bird
) Copsychus saularis Ilish ( national fish )
</item 8>

<item 9>
List of national animals
Jump up ^ Jon Eldridge . `` Primeval Poland : where the bison roam '' . theguardian.com . Retrieved 2016 - 10 - 10 . Jump up ^ `` Poland '' . allaboutbison.com
. Retrieved 2016 - 10 - 10 . Jump up ^ Tamra Orr ( 30 June 2008 ) . Qatar . Marshall Cavendish . p. 13 . ISBN 978 - 0 - 7614 - 2566 - 3 . Retrieved 30 July 2011
. Jump up ^ `` Animalul national - Rasul - Sapte simboluri nationale ale Romaniei mai putin cunoscute '' . 9AM . Jump up ^ `` Birobidjan , Russia Beaverton 's
sister city since 1990 '' . The City of Beaverton . Retrieved 2017 - 01 - 23 . Jump up ^ `` Rwanda State Symbols , Song , Flag , and More '' . worldatlas.com .
Retrieved 2013 - 03 - 21 .
</item 9>

<item 10>
List of national animals
Jump up ^ `` National Symbols of Pakistan '' . Government of Pakistan . Archived from the original on 27 August 2013 . Retrieved 23 August 2013 . Jump up ^
Hudson , Brydget E.T. ( 1981 ) . `` Interview and Aerial Survey Data in Relation to Resource Management of the Dugong in Manus Province , Papua New Guinea '' .
Bulletin of Marine Science . University of Miami . 31 ( 3 ) : 662 -- 672 . Retrieved 4 July 2011 . Jump up ^ `` Ten Unusual National Animals That Rival the
Unicorn - 9 of 10 '' . smithsonianmag.com . Retrieved 2014 - 10 - 03 . Jump up ^ Aquino , Dante M. ; Persoon , Gerald A. ( 2013 ) . `` Tradition and Change :
Beer Consumption in Northeast Luzon , Philippines '' . In Schiefenhovel , Wulf ; Macbeth , Helen . Liquid Bread : Beer and Brewing in Cross-Cultural Perspective
. Volume 7 of Anthropology of Food & Nutrition . Berghahn Books .
</item 10>
=== END ITEM id=53b74941d353abf0 ===

=== ITEM id=cbaa027b9748182c ===
Gold sentences and facts:
Sentence 0: Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa . Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order , an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star Wars : The Last Jedi ( 2017 ) .
  Fact 0.0: Kylo Ren is a fictional character in the Star Wars franchise.
  Fact 0.1: Kylo Ren was introduced in the 2015 film Star Wars: The Force Awakens.
  Fact 0.2: Adam Driver portrays Kylo Ren.
  Fact 0.3: Kylo Ren is the chosen name of Ben Solo.
  Fact 0.4: Ben Solo is the son of Han Solo and Leia Organa.
  Fact 0.5: Han Solo and Leia Organa are original Star Wars trilogy characters.
  Fact 0.6: Kylo Ren was trained as a Jedi by his uncle Luke Skywalker.
  Fact 0.7: Kylo Ren has been seduced to the dark side of the Force by Supreme Leader Snoke.
  Fact 0.8: Kylo Ren aspires to be as powerful as his grandfather, Darth Vader.
  Fact 0.9: Kylo Ren is the master of the Knights of Ren.
  Fact 0.10: Kylo Ren is a commander of the First Order.
  Fact 0.11: Kylo Ren is later the supreme leader of the First Order.
  Fact 0.12: The First Order is an organization spawned from the fallen Galactic Empire.
  Fact 0.13: Kylo Ren is featured in The Force Awakens media and merchandising.
  Fact 0.14: Kylo Ren appears in the film's sequel, Star Wars: The Last Jedi (2017).
Sentence 1: Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous . Melissa Leon of The Daily Beast describes Ren 's use of the Force as `` formidable '' , citing his ability to stop a blaster shot mid-air , immobilize victims and probe their minds against their will .
  Fact 1.0: Abrams told Empire in August 2015 that Kylo Ren is not a Sith.
  Fact 1.1: Abrams told Empire that Kylo Ren works under Supreme Leader Snoke.
  Fact 1.2: Abrams told Empire that Supreme Leader Snoke is a powerful figure on the Dark Side of the Force.
  Fact 1.3: Abrams had previously told Entertainment Weekly that the character came to the name Kylo Ren when he joined a group called the Knights of Ren.
  Fact 1.4: Robbie Collin of The Telegraph described Kylo Ren as a hot-headed, radicalised Dark Side jihadi.
  Fact 1.5: Robbie Collin described Kylo Ren's red lightsaber as spluttering and crackling as violently as his temper.
  Fact 1.6: Abrams noted that Kylo Ren built the lightsaber himself.
  Fact 1.7: Abrams noted that the lightsaber is as dangerous, fierce, and ragged as the character.
  Fact 1.8: The Telegraph explains that Kylo Ren's wild and erratic temper makes him dangerous.
  Fact 1.9: The Telegraph explains that Kylo Ren's angsty instability makes him dangerous.
  Fact 1.10: Melissa Leon of The Daily Beast describes Kylo Ren's use of the Force as formidable.
  Fact 1.11: Melissa Leon cites Kylo Ren's ability to stop a blaster shot mid-air.
  Fact 1.12: Melissa Leon cites Kylo Ren's ability to immobilize victims.
  Fact 1.13: Melissa Leon cites Kylo Ren's ability to probe victims' minds against their will.

Retrieved items:
<item 1>
Kylo Ren
You normally think of order , and structure , and full commitment and no hesitation ... he 's just a little bit more unpolished . It 's in his costume , in his
lightsaber -- how you kind of get the sense that it could just not work at any moment ; that it could just blow up . That 's kind of like a big metaphor for him
. Appearances ( edit ) Model of Kylo 's cross-bladed lightsaber on display at Star Wars Launch Bay at Disney 's Hollywood Studios ( 2015 ) . Feature films (
edit ) The Force Awakens ( 2015 ) ( edit ) Main article : Star Wars : The Force Awakens Kylo Ren first appears in The Force Awakens as the commander of the
First Order , a sinister regime that has risen from the remains of the old Galactic Empire . After arriving at Jakku to retrieve a map containing the
coordinates for the location of Luke Skywalker , Ren kills an old vicar known as Lor San Tekka and captures Resistance pilot Poe Dameron , who has also been
sent to recover the map by General Leia Organa .
</item 1>

<item 2>
Kylo Ren
Kylo Ren - wikipedia Kylo Ren Jump to : navigation , search `` Kylo '' redirects here . For the web browser , see Kylo ( web browser ) . Kylo Ren Ben Solo Star
Wars character Kylo Ren 's costume displayed on a mannequin First appearance The Force Awakens ( 2015 ) Created by Lawrence Kasdan J.J.
</item 2>

<item 3>
Kylo Ren
Before the duel is finished , they are separated by a seismic fissure created by the collapsing base . Rey and the others escape as Snoke orders General Hux to
evacuate the base and bring Ren to him to complete his training . The last Jedi ( 2017 ) ( edit ) Main article : Star Wars : The Last Jedi Ren 's inner conflict
continues into The Last Jedi , specifically through his conversations with Rey , with whom he connects through the Force . Rey believes that his conflict will
ultimately bring him back to the light side . Ren captures her and brings her before Snoke , who orders Ren to kill her . However , Ren uses the Force to ignite
a lightsaber at Snoke 's side and cut him in half , killing him . Ren beseeches Rey to join him in creating a new order , separate from the legacies created by
Snoke and Luke . When Rey refuses , Ren declares himself Supreme Leader of the First Order , and orders his forces to attack the Resistance base on Crait .
</item 3>

<item 4>
Kylo Ren
Though trained by his uncle Luke Skywalker as a Jedi , he has been seduced to the dark side of the Force by Supreme Leader Snoke and aspires to be as powerful
as his grandfather , Darth Vader . Kylo Ren is also the master of the Knights of Ren , as well as a commander and later the supreme leader of the First Order ,
an organization spawned from the fallen Galactic Empire . He is featured in The Force Awakens media and merchandising and appears in the film 's sequel , Star
Wars : The Last Jedi ( 2017 ) . Contents ( hide ) 1 Concept and creation 2 Character 3 Appearances 3.1 Feature films 3.1. 1 The Force Awakens ( 2015 ) 3.1. 2
The Last Jedi ( 2017 ) 3.2 Related works and promotional material 3.3 Novels 4 Reception 5 Family tree 6 References 7 External links Concept and creation ( edit
) After Lucasfilm was acquired by The Walt Disney Company in 2012 , the script for The Force Awakens was written by Lawrence Kasdan , J.J.
</item 4>

<item 5>
Kylo Ren
Ren engages Luke in battle and appears to blow him up and cut him in half , but Luke remains standing ; `` Luke '' is revealed to be simply a vision projected
from the real Luke on Ahch - To , serving as a distraction which allows the Resistance to escape from the First Order . When Luke vanishes , the First Order
storms the base , but Resistance has already evacuated ; he shares a final look with Rey through the Force before Rey closes the door to the Millennium Falcon
and escapes with the Resistance . Related works and promotional material ( edit ) He is a playable character in the 2015 Force Awakens add - on to the Disney
Infinity 3.0 video game , with an Infinity character figurine available separately . He is also a character in the strategy video game Star Wars : Force Arena .
Hasbro has released a 3 ⁄ - inch ( 9.5 cm ) Kylo Ren action figure , and a 6 - inch ( 15 cm ) figure in their Black Series line .
</item 5>

<item 6>
Kylo Ren
Abrams Michael Arndt Portrayed by Adam Driver Voiced by Adam Driver ( Disney Infinity 3.0 , Lego Star Wars : The Force Awakens , Lego Star Wars : The Resistance
Rises ) Matthew Wood ( Star Wars Battlefront II ) Roger Craig Smith ( Star Wars Battlefront II ) Information Full name Ben Solo Species Human Gender Male
Occupation Supreme Leader of the First Order Master of the Knights of Ren Former Jedi Padawan Affiliation First Order Knights of Ren New Jedi Order ( formerly )
Family Han Solo ( father ) Leia Organa ( mother ) Relatives Luke Skywalker ( maternal uncle ) Anakin Skywalker ( maternal grandfather ) Padmé Amidala ( maternal
grandmother ) Bail Organa ( adopted maternal grandfather ) Breha Organa ( adopted maternal grandmother ) Shmi Skywalker ( maternal great - grandmother )
Homeworld Chandrila Kylo Ren is a fictional character in the Star Wars franchise . Introduced in the 2015 film Star Wars : The Force Awakens , he is portrayed
by Adam Driver . Kylo Ren is the chosen name of Ben Solo , the son of original Star Wars trilogy characters Han Solo and Leia Organa .
</item 6>

<item 7>
Kylo Ren
Novels ( edit ) Kylo Ren is a point of view character in the 2015 novelization of The Force Awakens by Alan Dean Foster . In the 2017 Chuck Wendig novel Star
Wars : Aftermath : Empire 's End , Ben Solo is stated as having been born on the planet Chandrila on the same day as a peace treaty is signed between the
remnants of the Empire and the New Galactic Republic . Kylo Ren is additionally mentioned in the 2017 novel Star Wars : Phasma . Reception ( edit ) Adam Driver
received positive reviews for his performance as Kylo Ren . Both the character of Kylo Ren and Driver 's performance have been positively received . Many
complimented Ren 's conflicted nature and depth , as well as his costume design , and noted there were many places the character could be taken in future
installments . Peter Bradshaw of The Guardian praised the character and the actor alike , saying , `` He is gorgeously cruel , spiteful and capricious -- and
unlike the Vader of old , he is given to petulant temper tantrums , with his lightsaber drawn . ''
</item 7>

<item 8>
Kylo Ren
Abrams and Michael Arndt , with no direct participation from Star Wars creator George Lucas . According to director Abrams , the First Order is inspired by
ODESSA , a theoretical network of SS officers who fled to Argentina following World War II . Driver 's casting in the film in an unnamed role was first
announced on April 29 , 2014 . Kylo Ren was first seen , but still not named , in the 88 - second The Force Awakens teaser trailer released by Lucasfilm on
November 28 , 2014 , wielding a jagged red lightsaber with a crossguard . The name Kylo Ren was revealed by Entertainment Weekly in a Lucasfilm - designed Topps
- style trading card mockup in December 2014 . A May 2015 Vanity Fair photo shoot by Annie Leibovitz confirmed that Driver would be portraying Kylo . In an
interview with Time , The Force Awakens costume designer Michael Kaplan said that Abrams requested that Ren 's mask be designed to be memorable for a child .
After numerous attempts to gain Abrams ' approval , the final design was selected .
</item 8>

<item 9>
Kylo Ren
Ren 's character arc shares similarities with that of the Star Wars expanded universe character Jacen Solo , the son of Han Solo and Princess Leia who threatens
the galaxy as a fallen Jedi . Additionally , critics have noted a resemblance between Ren 's character design and that of Revan , the protagonist of Knights of
the Old Republic . Character ( edit ) Abrams told Empire in August 2015 , `` Kylo Ren is not a Sith . He works under Supreme Leader Snoke , who is a powerful
figure on the Dark Side of the Force . '' Abrams had previously told Entertainment Weekly that the character , `` came to the name Kylo Ren when he joined a
group called the Knights of Ren . '' Robbie Collin of The Telegraph described Ren as `` a hot - headed , radicalised Dark Side jihadi , whose red lightsaber
splutters and crackles as violently as his temper '' . Abrams noted , `` The lightsaber is something that he built himself , and is as dangerous and as fierce
and as ragged as the character . '' The Telegraph also explains that Ren 's wild and erratic temper and `` angsty '' instability make him dangerous .
</item 9>

<item 10>
Kylo Ren
Ren soon learns that the pilot had entrusted his astromech droid , BB - 8 , with the map . Poe flees with the help of rogue stormtrooper Finn , who later finds
BB - 8 , and the scavenger Rey . Finn , Rey , and BB - 8 escape Jakku in the Millennium Falcon , and are soon intercepted by the ship 's former owner , Han Solo
, and his co-pilot Chewbacca . It is revealed that Ren is the son of Han and Leia , and was once one of Luke 's Jedi pupils . He was drawn to the dark side of
the Force by the First Order 's Supreme Leader Snoke . He then helped to destroy his uncle Luke 's new Jedi Academy . However , Ren still feels the pull of the
light side of the Force and seeks the strength to overcome this weakness from his grandfather Darth Vader , whose burnt helmet is in Ren 's possession .
Arriving at the Maz Kanata castle , Ren captures Rey , who he senses has seen the map . While interrogating her with the Force , he realizes that she is strong
with the Force herself , though unaware of it .
</item 10>
=== END ITEM id=cbaa027b9748182c ===

=== ITEM id=a16c2865cd5ae3d2 ===
Gold sentences and facts:
Sentence 0: Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait * Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org
  Fact 0.0: OPEC's headquarters are in Vienna, Austria.
  Fact 0.1: OPEC's official language is English.
  Fact 0.2: OPEC is an international cartel.
  Fact 0.3: OPEC had 14 member states as of May 2017.
  Fact 0.4: OPEC's member states are Algeria, Angola, Ecuador, Equatorial Guinea, Gabon, Iran, Iraq, Kuwait, Libya, Nigeria, Qatar, Saudi Arabia, the United Arab Emirates, and Venezuela.
  Fact 0.5: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela are founding members of OPEC.
  Fact 0.6: OPEC's Secretary General is Mohammed Barkindo.
  Fact 0.7: OPEC was established in Baghdad, Iraq.
  Fact 0.8: OPEC's statute is dated September 1960.
  Fact 0.9: OPEC's statute came into effect in January 1961.
  Fact 0.10: OPEC's currency is indexed as USD per barrel (US$/bbl).
  Fact 0.11: OPEC's website is OPEC.org.
Sentence 1: Organization of the Petroleum Exporting Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria . As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies .
  Fact 1.0: OPEC is also called OPEP in several other languages.
  Fact 1.1: OPEC is pronounced OH-pek.
  Fact 1.2: OPEC is an intergovernmental organization of 14 nations as of February 2018.
  Fact 1.3: OPEC was founded in 1960 in Baghdad.
  Fact 1.4: OPEC was founded by the first five members: Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela.
  Fact 1.5: OPEC has been headquartered in Vienna, Austria since 1965.
  Fact 1.6: As of 2016, OPEC's 14 countries accounted for an estimated 44 percent of global oil production.
  Fact 1.7: As of 2016, OPEC's 14 countries held an estimated 73 percent of the world's 'proven' oil reserves.
  Fact 1.8: OPEC has a major influence on global oil prices.
  Fact 1.9: Global oil prices were previously determined by American-dominated multinational oil companies.
Sentence 2: Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa 1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 -- 1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa 1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,000 Iraq Middle East 1960 -- 37,202,572 437,072 4,451,516 143,069,000,000 Kuwait Middle East 1960 -- 4,052,584 17,820 2,923,825 101,500,000,000 Libya North Africa 1962 -- 6,293,253 1,759,540 384,686 48,363,000,000 Nigeria West Africa 1971 -- 185,989,640 923,768 1,999,885 37,070,000,000 Qatar Middle East 1961 -- 2,569,804 11,437 1,522,902 25,244,000,000 Saudi Arabia Middle East 1960 -- 32,275,687 2,149,690 10,460,710 266,578,000,000 United Arab Emirates Middle East 1967 -- 9,269,612 83,600 3,106,077 97,800,000,000 Venezuela South America 1960 -- 31,568,179 912,050 2,276,967 299,953,000,000 OPEC Total 478,498,000 12,150,695 35,221,740 1,209,103,000,000 World Total 7,618,511,000 510,072,000 80,622,287 1,650,585,000,000 OPEC Percent 6 % 2 % 44 % 73 %
  Fact 2.0: Algeria is in North Africa, joined OPEC in 1969, has a population of 40,606,052, an area of 2,381,740 km, oil production of 1,348,361 bbl/day in 2016, and proven reserves of 12,200,000,000 bbl in 2016.
  Fact 2.1: Angola is in Southern Africa, joined OPEC in 2007, has a population of 28,813,463, an area of 1,246,700 km, oil production of 1,769,615 bbl/day in 2016, and proven reserves of 8,423,000,000 bbl in 2016.
  Fact 2.2: Ecuador is in South America, was a member in 1973 to 1992 and rejoined in 2007, has a population of 16,385,068, an area of 283,560 km, oil production of 548,421 bbl/day in 2016, and proven reserves of 8,273,000,000 bbl in 2016.
  Fact 2.3: Equatorial Guinea is in Central Africa, joined OPEC in 2017, has a population of 1,221,490, an area of 28,051 km, oil production of 227,000 bbl/day in 2016, and proven reserves of 1,100,000,000 bbl in 2016.
  Fact 2.4: Gabon is in Central Africa, was a member from 1975 to 1995 and rejoined in 2016, has a population of 1,979,786, an area of 267,667 km, oil production of 210,820 bbl/day in 2016, and proven reserves of 2,000,000,000 bbl in 2016.
  Fact 2.5: Iran is in the Middle East, joined OPEC in 1960, has a population of 80,277,428, an area of 1,648,000 km, oil production of 3,990,956 bbl/day in 2016, and proven reserves of 157,530,000,000 bbl in 2016.
  Fact 2.6: Iraq is in the Middle East, joined OPEC in 1960, has a population of 37,202,572, an area of 437,072 km, oil production of 4,451,516 bbl/day in 2016, and proven reserves of 143,069,000,000 bbl in 2016.
  Fact 2.7: Kuwait is in the Middle East, joined OPEC in 1960, has a population of 4,052,584, an area of 17,820 km, oil production of 2,923,825 bbl/day in 2016, and proven reserves of 101,500,000,000 bbl in 2016.
  Fact 2.8: Libya is in North Africa, joined OPEC in 1962, has a population of 6,293,253, an area of 1,759,540 km, oil production of 384,686 bbl/day in 2016, and proven reserves of 48,363,000,000 bbl in 2016.
  Fact 2.9: Nigeria is in West Africa, joined OPEC in 1971, has a population of 185,989,640, an area of 923,768 km, oil production of 1,999,885 bbl/day in 2016, and proven reserves of 37,070,000,000 bbl in 2016.
  Fact 2.10: Qatar is in the Middle East, joined OPEC in 1961, has a population of 2,569,804, an area of 11,437 km, oil production of 1,522,902 bbl/day in 2016, and proven reserves of 25,244,000,000 bbl in 2016.
  Fact 2.11: Saudi Arabia is in the Middle East, joined OPEC in 1960, has a population of 32,275,687, an area of 2,149,690 km, oil production of 10,460,710 bbl/day in 2016, and proven reserves of 266,578,000,000 bbl in 2016.
  Fact 2.12: The United Arab Emirates is in the Middle East, joined OPEC in 1967, has a population of 9,269,612, an area of 83,600 km, oil production of 3,106,077 bbl/day in 2016, and proven reserves of 97,800,000,000 bbl in 2016.
  Fact 2.13: Venezuela is in South America, joined OPEC in 1960, has a population of 31,568,179, an area of 912,050 km, oil production of 2,276,967 bbl/day in 2016, and proven reserves of 299,953,000,000 bbl in 2016.
  Fact 2.14: OPEC's total for 2016 is a population of 478,498,000, an area of 12,150,695 km, oil production of 35,221,740 bbl/day, and proven reserves of 1,209,103,000,000 bbl.
  Fact 2.15: The world's total for 2016 is a population of 7,618,511,000, an area of 510,072,000 km, oil production of 80,622,287 bbl/day, and proven reserves of 1,650,585,000,000 bbl.
  Fact 2.16: OPEC's share for 2016 is 6 percent of world population, 2 percent of world area, 44 percent of world oil production, and 73 percent of world proven reserves.

Retrieved items:
<item 1>
OPEC
269,612 83,600 3,106,077 97,800,000,000 Venezuela South America 1960 -- 31,568,179 912,050 2,276,967 299,953,000,000 OPEC Total 478,498,000 12,150,695
35,221,740 1,209,103,000,000 World Total 7,618,511,000 510,072,000 80,622,287 1,650,585,000,000 OPEC Percent 6 % 2 % 44 % 73 % ^ Jump up to : One petroleum
barrel ( bbl ) is approximately 42 U.S. gallons , or 159 liters , or 0.159 m , varying slightly with temperature . To put the production numbers in context , a
supertanker typically holds 2,000,000 barrels ( 320,000 m ) , and the world 's current production rate would take approximately 56 years to exhaust the world 's
current proven reserves . ^ Jump up to : The five founding members attended the first OPEC conference in September 1960 .
</item 1>

<item 2>
OPEC
As of 2016 , the 14 countries accounted for an estimated 44 percent of global oil production and 73 percent of the world 's `` proven '' oil reserves , giving
OPEC a major influence on global oil prices that were previously determined by American - dominated multinational oil companies . OPEC 's stated mission is ``
to coordinate and unify the petroleum policies of its member countries and ensure the stabilization of oil markets , in order to secure an efficient , economic
and regular supply of petroleum to consumers , a steady income to producers , and a fair return on capital for those investing in the petroleum industry . ''
The organization is also a significant provider of information about the international oil market . As of May 2017 , OPEC 's members are Algeria , Angola ,
Ecuador , Equatorial Guinea , Gabon , Iran , Iraq , Kuwait , Libya , Nigeria , Qatar , Saudi Arabia ( the de facto leader ) , United Arab Emirates , and
Venezuela , while Indonesia is a former member . Two - thirds of OPEC 's oil production and reserves are in its six Middle Eastern countries that surround the
oil - rich Persian Gulf .
</item 2>

<item 3>
OPEC
However , their influence on international trade is periodically challenged by the expansion of non-OPEC energy sources , and by the recurring temptation for
individual OPEC countries to exceed production ceilings and pursue conflicting self - interests . Contents ( hide ) 1 Membership 1.1 Current member countries
1.2 Lapsed members 1.3 Observers 1.4 Leadership and decision - making 1.5 International cartel 1.6 Conflicts 2 Market information 2.1 Publications and research
2.2 Crude oil benchmarks 2.3 Spare capacity 3 History and impact 3.1 Post-WWII situation 3.2 1959 -- 1960 anger from exporting countries 3.3 1960 -- 1975
founding and expansion 3.4 1973 -- 1974 oil embargo 3.5 1975 -- 1980 Special Fund , now OFID 3.6 1975 hostage siege 3.7 1979 -- 1980 oil crisis and 1980s oil
glut 3.8 1990 -- 2003 ample supply and modest disruptions 3.9 2003 -- 2011 volatility 3.
</item 3>

<item 4>
OPEC
During 1961 -- 1975 , the five founding nations were joined by Qatar ( 1961 ) , Indonesia ( 1962 -- 2008 , rejoined 2014 - 2016 ) , Libya ( 1962 ) , United Arab
Emirates ( originally just the Emirate of Abu Dhabi , 1967 ) , Algeria ( 1969 ) , Nigeria ( 1971 ) , Ecuador ( 1973 -- 1992 , rejoined 2007 ) , and Gabon ( 1975
-- 1994 , rejoined 2016 ) . By the early 1970s , OPEC 's membership accounted for more than half of worldwide oil production . Indicating that OPEC is not
averse to further expansion , Mohammed Barkindo , OPEC 's Acting Secretary General in 2006 , urged his African neighbors Angola and Sudan to join , and Angola
did in 2007 , followed by Equatorial Guinea in 2017 . Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia , and other oil - exporting
nations have attended many OPEC meetings as observers , as an informal mechanism for coordinating policies .
</item 4>

<item 5>
OPEC
In May 2008 , Indonesia announced that it would leave OPEC when its membership expired at the end of that year , having become a net importer of oil and being
unable to meet its production quota . It rejoined the organization in January 2016 , but announced another `` temporary suspension '' of its membership at year
- end when OPEC requested a 5 percent production cut . Some commentators consider that the United States was a de facto member of OPEC during its formal
occupation of Iraq , due to its leadership of the Coalition Provisional Authority in 2003 -- 2004 . But this is not borne out by the minutes of OPEC meetings ,
as no US representative attended in an official capacity . Observers ( edit ) Since the 1980s , representatives from Egypt , Mexico , Norway , Oman , Russia ,
and other oil - exporting nations have attended many OPEC meetings as observers . This arrangement serves as an informal mechanism for coordinating policies .
</item 5>

<item 6>
OPEC
Country Region Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Algeria North Africa
1969 -- 40,606,052 2,381,740 1,348,361 12,200,000,000 Angola Southern Africa 2007 -- 28,813,463 1,246,700 1,769,615 8,423,000,000 Ecuador South America 1973 --
1992 , 2007 -- 16,385,068 283,560 548,421 8,273,000,000 Equatorial Guinea Central Africa 2017 -- 1,221,490 28,051 227,000 1,100,000,000 Gabon Central Africa
1975 -- 1995 , 2016 -- 1,979,786 267,667 210,820 2,000,000,000 Iran Middle East 1960 -- 80,277,428 1,648,000 3,990,956 157,530,000,
</item 6>

<item 7>
OPEC
Jump up ^ The UAE was founded in December 1971 . Its OPEC membership originated with the Emirate of Abu Dhabi . Lapsed members ( edit ) Country Region
Membership Years Population ( 2016 est . ) Area ( km ) Oil Production ( bbl / day , 2016 ) Proven Reserves ( bbl , 2016 ) Indonesia Southeast Asia 1962 -- 2008
, Jan - Nov 2016 261,115,456 1,904,569 833,667 3,692,500,000 For countries that export petroleum at relatively low volume , their limited negotiating power as
OPEC members would not necessarily justify the burdens imposed by OPEC production quotas and membership costs . Ecuador withdrew from OPEC in December 1992 ,
because it was unwilling to pay the annual US $2 million membership fee and felt that it needed to produce more oil than it was allowed under its OPEC quota at
the time , although it rejoined in October 2007 . Similar concerns prompted Gabon to suspend membership in January 1995 ; it rejoined in July 2016 .
</item 7>

<item 8>
OPEC
OPEC - wikipedia OPEC Jump to : navigation , search Organization of the Petroleum Exporting Countries ( OPEC ) Flag Headquarters Vienna , Austria Official
language English Type International cartel Membership 14 states ( May 2017 ) ( show ) Algeria Angola Ecuador Equatorial Guinea Gabon Iran * Iraq * Kuwait *
Libya Nigeria Qatar Saudi Arabia * United Arab Emirates Venezuela * * founding member Leaders Secretary General Mohammed Barkindo Establishment Baghdad , Iraq
Statute September 1960 In effect January 1961 Currency Indexed as USD per barrel ( US $ / bbl ) Website OPEC.org Organization of the Petroleum Exporting
Countries ( OPEC , / ˈoʊpɛk / OH - pek , or OPEP in several other languages ) is an intergovernmental organization of 14 nations as of February 2018 , founded
in 1960 in Baghdad by the first five members ( Iran , Iraq , Kuwait , Saudi Arabia , and Venezuela ) , and headquartered since 1965 in Vienna , Austria .
</item 8>

<item 9>
OPEC
9 2003 -- 2011 volatility 3.10 2008 production dispute 3.11 2014 -- 2017 oil glut 3.12 2017 - 2018 Production cut 4 See also 5 References 6 External links
Membership ( edit ) Current member countries ( edit ) As of May 2017 , OPEC has 14 member countries : six in the Middle East ( Western Asia ) , six in Africa ,
and two in South America . According to the U.S. Energy Information Administration ( EIA ) , OPEC 's combined rate of oil production ( including gas condensate
) represented 44 percent of the world 's total in 2016 , and OPEC accounted for 73 percent of the world 's `` proven '' oil reserves , including 48 percent from
just the six Middle Eastern members : Approval of a new member country requires agreement by three - quarters of OPEC 's existing members , including all five
of the founders . In October 2015 , Sudan formally submitted an application to join , but it is not yet a member . Country Region Membership Years Population (
2016 est . )
</item 9>

<item 10>
OPEC
A statement released by OPEC on 10 September 2008 confirmed Indonesia 's withdrawal , noting that OPEC `` regretfully accepted the wish of Indonesia to suspend
its full membership in the organization , and recorded its hope that the country would be in a position to rejoin the organization in the not - too - distant
future . '' 2008 production dispute ( edit ) Countries by net oil exports ( 2008 ) The differing economic needs of OPEC member states often affect the internal
debates behind OPEC production quotas . Poorer members have pushed for production cuts from fellow members , to increase the price of oil and thus their own
revenues . These proposals conflict with Saudi Arabia 's stated long - term strategy of being a partner with the world 's economic powers to ensure a steady
flow of oil that would support economic expansion . Part of the basis for this policy is the Saudi concern that overly expensive oil or unreliable supply will
drive industrial nations to conserve energy and develop alternative fuels , curtailing the worldwide demand for oil and eventually leaving unneeded barrels in
the ground . To this point , Saudi Oil Minister Yamani famously remarked in 1973 : `` The Stone Age did n't end because we ran out of stones . ''
</item 10>
=== END ITEM id=a16c2865cd5ae3d2 ===

=== ITEM id=d4de7668924effc5 ===
Gold sentences and facts:
Sentence 0: Prior to 1940 the term `` jeep '' had been used as U.S. Army slang for new recruits or vehicles , but the World War II `` jeep '' that went into production in 1941 specifically tied the name to this light military 4x4 , arguably making them the oldest four - wheel drive mass - production vehicles now known as SUVs . The Jeep became the primary light 4 - wheel - drive vehicle of the United States Army and the Allies during World War II , as well as the postwar period . The term became common worldwide in the wake of the war . Doug Stewart noted : `` The spartan , cramped , and unstintingly functional jeep became the ubiquitous World War II four - wheeled personification of Yankee ingenuity and cocky , can - do determination . ''
  Fact 0.0: Before 1940, the term 'jeep' was used as U.S. Army slang for new recruits or vehicles.
  Fact 0.1: The World War II 'jeep' went into production in 1941.
  Fact 0.2: The World War II 'jeep' specifically tied the name 'jeep' to a light military 4x4.
  Fact 0.3: The World War II jeep arguably made the jeep one of the oldest four-wheel-drive mass-production vehicles now known as SUVs.
  Fact 0.4: The Jeep became the primary light 4-wheel-drive vehicle of the United States Army during World War II.
  Fact 0.5: The Jeep became the primary light 4-wheel-drive vehicle of the Allies during World War II.
  Fact 0.6: The Jeep remained the primary light 4-wheel-drive vehicle of the United States Army and the Allies in the postwar period.
  Fact 0.7: The term 'jeep' became common worldwide in the wake of the war.
  Fact 0.8: Doug Stewart noted that the jeep was spartan, cramped, and unstintingly functional.
  Fact 0.9: Doug Stewart noted that the jeep became the ubiquitous World War II four-wheeled personification of Yankee ingenuity.
  Fact 0.10: Doug Stewart noted that the jeep became the personification of cocky, can-do determination.
Sentence 1: At least two Jeep models ( the CJ - 5 and the SJ Wagoneer ) enjoyed extraordinary three - decade production runs of a single body generation . Jeeps have since the war inspired a number of other light utility vehicles , such as the Land Rover . Many Jeep variants serving similar military and civilian roles have since been designed in other nations . In lowercase the term `` jeep '' continues to be used as a generic term for vehicles inspired by the Jeep that are suitable for use on rough terrain .
  Fact 1.0: At least two Jeep models, the CJ-5 and the SJ Wagoneer, enjoyed extraordinary three-decade production runs of a single body generation.
  Fact 1.1: Jeeps have inspired a number of other light utility vehicles since the war, such as the Land Rover.
  Fact 1.2: Many Jeep variants serving similar military and civilian roles have been designed in other nations since the war.
  Fact 1.3: In lowercase, the term 'jeep' continues to be used as a generic term for vehicles inspired by the Jeep that are suitable for use on rough terrain.
Sentence 2: Jeep 's current product range consists solely of sport utility vehicles and off - road vehicles , but has also included pickup trucks in the past . Some of Jeep 's vehicles , such as the Grand Cherokee and the Wagoneer , reach into the luxury segment . Jeep sold 1.4 million SUVs globally in 2016 , up from 500,000 in 2008 , two - thirds of which in North America , and was Fiat - Chrysler 's best selling brand in the U.S. during the first half of 2017 . In the U.S. alone , over 2400 dealerships hold franchise rights to sell Jeep - branded vehicles , and if Jeep were spun off into a separate company , it is estimated to be worth between $22 and $33.5 billion -- slightly more than all of FCA ( US ) .
  Fact 2.0: Jeep's current product range consists solely of sport utility vehicles and off-road vehicles.
  Fact 2.1: Jeep's product range has also included pickup trucks in the past.
  Fact 2.2: Some Jeep vehicles, such as the Grand Cherokee and the Wagoneer, reach into the luxury segment.
  Fact 2.3: Jeep sold 1.4 million SUVs globally in 2016, up from 500,000 in 2008.
  Fact 2.4: Two-thirds of Jeep's 1.4 million SUVs sold globally in 2016 were sold in North America.
  Fact 2.5: Jeep was Fiat-Chrysler's best selling brand in the U.S. during the first half of 2017.
  Fact 2.6: Over 2400 dealerships in the U.S. hold franchise rights to sell Jeep-branded vehicles.
  Fact 2.7: If Jeep were spun off into a separate company, it is estimated to be worth between $22 and $33.5 billion.
  Fact 2.8: The estimated worth of a spun-off Jeep is slightly more than all of FCA (US).

Retrieved items:
<item 1>
Jeep
With its solid axles and open top , the Wrangler has been called the Jeep model that is as central to the brand 's identity as the rear - engined 911 is to
Porsche . At least two Jeep models ( the CJ - 5 and the SJ Wagoneer ) enjoyed extraordinary three - decade production runs of a single body generation . Jeeps
have since the war inspired a number of other light utility vehicles , such as the Land Rover . Many Jeep variants serving similar military and civilian roles
have since been designed in other nations . In lowercase the term `` jeep '' continues to be used as a generic term for vehicles inspired by the Jeep that are
suitable for use on rough terrain . Contents ( hide ) 1 World War II Jeeps 1.1 Development -- 1 . Bantam Reconnaissance Car 1.2 Development -- 2 . Enter Willys
and Ford 1.3 Full production -- Willys MB and Ford GPW 2 Post-war military Jeeps 2.1 CJ - V35 / U 2.2 M715 3 Jeep etymology 4 Brand , trademarks and image 4.1
Off - road abilities 5 Company history and ownership 5.
</item 1>

<item 2>
Jeep
5 Main articles : Jeep Cherokee ( XJ ) and Jeep Comanche 1984 -- 2001 XJ Cherokee 1984 -- 1990 XJ Wagoneer 1984 -- 1985 Broughwood 1984 -- 1990 Limited 1986 --
1992 Jeep Comanche ( MJ ) Jeep Wrangler 1989 Jeep Wrangler YJ Islander 1997 Jeep Wrangler TJ 2016 Jeep Wrangler JK Unlimited ( MIAS ' 16 ) Jeep Wrangler Main
article : Jeep Wrangler 1987 -- 1995 Jeep Wrangler YJ 1991 -- 1993 Renegade 1988 -- 1995 Wrangler Long -- Venezuela 1995 Wrangler Rio Grande 1997 -- 2006
Wrangler TJ 2002 TJ Se , X , Sport , Sahara models 2003 TJ Rubicon , Rubicon Tomb Raider Edition , Sahara , Sport , X , Se models , Freedom Edition 2004 -- 2006
TJ Long Wheel Base ( LJ ) Unlimited ( 15 '' Longer than a standard TJ ) Rubicon , Sport , X , Se models 2004 -- 2005 Willys Edition ( 2004 -- 1997 made ,
</item 2>

<item 3>
Jeep
Jeep 's current product range consists solely of sport utility vehicles and off - road vehicles , but has also included pickup trucks in the past . Some of Jeep
's vehicles , such as the Grand Cherokee and the Wagoneer , reach into the luxury segment . Jeep sold 1.4 million SUVs globally in 2016 , up from 500,000 in
2008 , two - thirds of which in North America , and was Fiat - Chrysler 's best selling brand in the U.S. during the first half of 2017 . In the U.S. alone ,
over 2400 dealerships hold franchise rights to sell Jeep - branded vehicles , and if Jeep were spun off into a separate company , it is estimated to be worth
between $22 and $33.5 billion -- slightly more than all of FCA ( US ) . Prior to 1940 the term `` jeep '' had been used as U.S.
</item 3>

<item 4>
Jeep
Also referred to as `` any small plane , helicopter , or gadget . '' This definition is supported by the use of the term `` jeep carrier '' to refer to the Navy
's small escort carriers . Early in 1941 , Willys - Overland demonstrated the vehicle 's off - road capability by having it drive up the steps of the United
States Capitol , driven by Willys test driver Irving `` Red '' Hausmann , who had recently heard soldiers at Fort Holabird calling it a `` jeep . '' When asked
by syndicated columnist Katharine Hillyer for the Washington Daily News ( or by a bystander , according to another account ) what it was called , Hausmann
answered , `` It 's a jeep . '' Katharine Hillyer 's article was published nationally on February 19 , 1941 , and included a picture of the vehicle with the
caption : LAWMAKERS TAKE A RIDE - With Senator Meade , of New York , at the wheel , and Representative Thomas , of New Jersey , sitting beside him , one of the
Army 's new scout cars , known as `` jeeps '' or `` quads '' , climbs up the Capitol steps in a demonstration yesterday .
</item 4>

<item 5>
Jeep
( prototype ) 1963 Jeep XM - 200 : J200 based concept for developing a low - cost vehicle for third - world countries 1965 Jeep / Renault Model H : A light 4x4
prototype based on the Renault 16 1966 FWD Concept Jeepvair : Similar to the Model H but with a Chevrolet Corvair powertrain 1970 XJ001 1970 XJ002 1971 Jeep
Cowboy : A design study using AMC 's `` compact '' automobile platform 1977 Jeep II 1979 Jeep Jeepster II 1986 Cherokee Targa : A two - door Cherokee
convertible ( later revised as Jeep Freedom show car ) 1987 Comanche Thunderchief : This vehicle was put into production later as the Comanche Eliminator 1989
Jeep Concept 1 : Evolved into the ZJ Grand Cherokee 1989 Jeep Rubicon Wrangler : This vehicle was later put in production 1990 Jeep JJ : Essentially what would
later be called the Icon 1990 Jeep Freedom : A revised Cherokee Targa 1991 Jeep Wagoneer 2000 : A large design concept 1993 Jeep Ecco 1997 Jeep Cherokee
Casablanca : A special edition of Cherokee ,
</item 5>

<item 6>
Jeep
windowless hardtop 2014 Jeep Wrangler Level Red 2014 Jeep Cherokee Dakar 2014 Jeep Wrangler MOJO 2015 Jeep Chief 2015 Jeep Wrangler Africa 2015 Jeep Wrangler
Red Rock Responder 2015 Jeep Staff Car : a tribute to Jeep 's military history starting with WWII Current models The Jeep brand currently produces five models ,
but 8 vehicles are under the brand name or use the Jeep logo : Jeep Renegade BU : Subcompact Sport Utility Vehicle Jeep Wrangler JK : Standard wheelbase Compact
Sport utility vehicle , 2 - door version JK Unlimited : Long wheelbase Mid-Size sport utility vehicle , 4 - door version J8 : Mid-Size military sport utility
vehicle ; Produced by AIL , AAV , and AEV . TJL : Compact pickup truck , 2 - door version ; Produced by AAV .
</item 6>

<item 7>
Jeep
Please consider summarizing the material while citing sources as needed . ( December 2017 ) Main article : List of Jeep vehicles Jeep CJ Main article : Jeep CJ
( Civilian Jeep ) 1982 Jeep Scrambler The CJ ( for `` Civilian Jeep '' ) series were literally the first `` Jeep '' branded vehicles produced for the civilian
public , beginning in 1945 with the CJ - 2A , followed by the CJ - 3A in 1949 and the CJ - 3B in 1953 . These early Jeeps are commonly referred to as ``
flatfenders '' because their front fenders were flat and rectangular across , like on their military forebears , the Willys MB and identical Ford GPW models .
The CJ - 4 exists only as a 1951 prototype , and is the `` missing link '' between the flat - fendered CJ - 2A and CJ - 3B and the round - fendered CJ - 5 first
introduced in 1955 .
</item 7>

<item 8>
Jeep
Please consider summarizing the material while citing sources as needed . ( April 2018 ) 1944 CJ - 1 prototype 1949 Alcoa Aluminum - bodied Jeepster Coupe (
prototype ) 1949 / 1950 X-98 prototype ; with flat fenders , but a rounded hood and grille like the CJ - 5 , it may have been the first F - head - powered Jeep
1950 CJ - 4 prototype 1950 CJ - 4 M prototype 1950 CJ - 4 MA prototype 1952 CJ Coiler : experimental design for an all independent suspension , with portal -
hub swing - axles and coil - springs 1958 DJ - 3A Pickup : Prototype pickup truck version of the DJ - 3A 1958 Jeep Creep : prototype utility vehicle ; several
versions built for tests , including a Postal rig and an aircraft tug 1959 Jeep J - 100 Malibu and Berkeley : Later developed into the Wagoneer 1960 Jeep Wide -
Trac : Concept for developing a low - cost vehicle for third - world countries 1962 The Brazilian Jeepster ( prototype ) 1963 Jeep XM - 200 : J200 based concept
for developing a
</item 8>

<item 9>
Jeep
2 - door version ; produced by AEV with the Jeep logo . 2007 -- 2018 Wrangler JK 2007 -- 2009 JK Rubicon , Sahara , X 2010 JK Rubicon , Sahara , Mountain ,
Islander , and Sport Editions 2011 70th Anniversary , Call of Duty : Black Ops , Mojave , and Oscar Mike Military ( 200 made ) Editions 2011 JK - 8 Independence
-- a MOPAR JK pick - up conversion kit , mirroring the 1980s CJ - 8 Scrambler pick - up 2013 Rubicon 10th Anniversary Edition 2013 - 2017 Brute Double Cab :
Pickup truck , 4 - door version , produced by American Expedition Vehicles 2014 Willys Wheeler Edition 2017 -- Jeep Wrangler JL Grand Cherokee Main article :
Jeep Grand Cherokee 1st generation Grand Cherokee ZJ . 2008 - 2010 WK Grand Cherokee .
</item 9>

<item 10>
Jeep
Jeep - wikipedia Jeep Jump to : navigation , search This article is about the U.S. motor manufacturing company . For the original World War II Jeep , see Willys
MB . For other uses , see Jeep ( disambiguation ) . Jeep Type Division Industry Automotive Founded 1943 trademark application 1945 first civilian Jeep launched
Headquarters Toledo , Ohio , U.S. Area served Worldwide Key people Michael Manley ( CEO of Jeep division ) Sergio Marchionne ( CEO of FCA US LLC ) Products
Sport utility vehicles Luxury vehicles Owner Fiat Chrysler Automobiles Parent Willys - Overland ( 1943 - 1953 ) Kaiser - Jeep ( 1953 - 1970 ) American Motors
Corporation ( 1970 - 1986 ) Chrysler ( 1987 - present ) Website jeep.com Jeep is a brand of American automobiles that is a division of FCA US LLC ( formerly
Chrysler Group , LLC ) , a wholly owned subsidiary of Fiat Chrysler Automobiles . Jeep has been a part of Chrysler since 1987 , when Chrysler acquired the Jeep
brand , along with remaining assets , from its previous owner : American Motors Corporation ( AMC ) .
</item 10>
=== END ITEM id=d4de7668924effc5 ===

=== ITEM id=ce64366cbfe9e4c9 ===
Gold sentences and facts:
Sentence 0: The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of the Mahanadi Delta .
  Fact 0.0: The Eastern Coastal Plains refers to a wide stretch of landmass of India.
  Fact 0.1: The Eastern Coastal Plains lie between the Eastern Ghats and the Bay of Bengal.
  Fact 0.2: The Eastern Coastal Plains are wider than the western coastal plains.
  Fact 0.3: The Eastern Coastal Plains are more leveled than the western coastal plains.
  Fact 0.4: The Eastern Coastal Plains stretch from Tamil Nadu in the south to West Bengal in the north.
  Fact 0.5: The Eastern Coastal Plains pass through Andhra Pradesh and Odisha.
  Fact 0.6: Chilka Lake is a brackish water lake along the eastern coastal plain.
  Fact 0.7: Chilka Lake lies in the state of Odisha.
  Fact 0.8: Chilka Lake stretches to the south of the Mahanadi Delta.

Retrieved items:
<item 1>
Eastern coastal plains
The width of the plains varies between 100 and 130 km ( 62 to 80 miles ) . It is locally known as Utkal Plains in the Northern part between Cossye and
Rushikulya Rivers , Northern Circars in the Central part between Rushikulya and Krishna Rivers and , as Coromandel Coast in the Southern part from the south of
river Krishna till the Southern tip of Mainland India at Cape Comorin where it merges with the Western Coastal Plains . See also ( edit ) Western Coastal Plains
References ( edit ) Jump up ^ Sr. Bimcy ; Sr. Sisily ; Charlotte . Bibliographic information . Scholar Publishing House . pp. 20 -- 21 . ISBN 8171725163 .
</item 1>

<item 2>
Eastern coastal plains
Eastern coastal plains - Wikipedia Eastern coastal plains Jump to : navigation , search This article relies largely or entirely on a single source . Relevant
discussion may be found on the talk page . Please help improve this article by introducing citations to additional sources . ( May 2015 ) View of Fields at
Biccavolu , Eastern coastal plains , Andhra Pradesh The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats
and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through
Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of
the Mahanadi Delta . Deltas of many of India 's rivers form a major portion of these plains . The Mahanadi , Godavari , Kaveri and Krishna rivers drain these
plains . The region receives both the Northeast & Southwest monsoon rains with its annual rainfall averaging between 1,000 and 3,000 mm ( 39 and 118 in ) .
</item 2>

<item 3>
Eastern coastal plains
pp. 20 -- 21 . ISBN 8171725163 . Geography of India Climate Climate Climatic regions Geology Fossil Parks Geology of India Indian Plate Stones Landforms Beaches
Canals Desert Extreme points Glaciers Islands Lakes Mountains Rivers Valleys Volcanoes Waterfalls Plains Indo - Gangetic Eastern Coastal Western Coastal Regions
North India Northeast India East India South India West India Central India Subdivisions Autonomous administrative divisions States and territories Districts
Cities Towns Municipalities Environment Biogeographic classification Ecoregions Fauna Flora Forests Issues Protected Areas Wildlife This Indian location article
is a stub . You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Eastern_coastal_plains&oldid=812520937 ''
Categories : Plains of India Coasts of India India geography stubs Hidden categories : Articles needing additional references from May 2015 All articles needing
additional references India articles missing geocoordinate data All articles needing coordinates All stub articles Talk About Wikipedia हिन्दी Português தமிழ்
Edit links This page was last edited on 28 November 2017 , at 08 : 32 .
</item 3>

<item 4>
Eastern coastal plains
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>
=== END ITEM id=ce64366cbfe9e4c9 ===
