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

This file has 5 items.

=== ITEM id=6731f7b474590e90 ===
Gold sentences and facts:
Sentence 0: Due to the high salinity , very few fish species can tolerate living in the Salton Sea . Introduced tilapia are the main fish that can tolerate the high salinity levels and pollution . Other freshwater fish species live in the rivers and canals that feed the Salton Sea , including threadfin shad , carp , red shiner , channel catfish , white catfish , largemouth bass , mosquitofish , sailfin molly , and the vulnerable desert pupfish .
  Fact 0.0: Due to the high salinity, very few fish species can tolerate living in the Salton Sea.
  Fact 0.1: Introduced tilapia are the main fish that can tolerate the high salinity levels and pollution.
  Fact 0.2: Other freshwater fish species live in the rivers and canals that feed the Salton Sea.
  Fact 0.3: Threadfin shad live in the rivers and canals that feed the Salton Sea.
  Fact 0.4: Carp live in the rivers and canals that feed the Salton Sea.
  Fact 0.5: Red shiner live in the rivers and canals that feed the Salton Sea.
  Fact 0.6: Channel catfish live in the rivers and canals that feed the Salton Sea.
  Fact 0.7: White catfish live in the rivers and canals that feed the Salton Sea.
  Fact 0.8: Largemouth bass live in the rivers and canals that feed the Salton Sea.
  Fact 0.9: Mosquitofish live in the rivers and canals that feed the Salton Sea.
  Fact 0.10: Sailfin molly live in the rivers and canals that feed the Salton Sea.
  Fact 0.11: Desert pupfish live in the rivers and canals that feed the Salton Sea.
  Fact 0.12: Desert pupfish are vulnerable.

Retrieved items:
<item 1>
Salton Sea
depth 43 ft ( 13 m ) Water volume 6,000,000 acre ⋅ ft ( 7.4 km ) Surface elevation − 236 ft ( − 71.9 m ) ( below sea level ) Settlements Bombay Beach , Desert
Beach , Desert Shores , Salton City , Salton Sea Beach , North Shore References U.S. Geological Survey Geographic Names Information System : Salton Sea Map of
the Salton Sea drainage area The Salton Sea is a shallow , saline , endorheic rift lake located directly on the San Andreas Fault , predominantly in California
's Imperial and Coachella valleys . The lake occupies the lowest elevations of the Salton Sink in the Colorado Desert of Imperial and Riverside Counties in
Southern California . Its surface is 236.0 ft ( 71.9 m ) below sea level as of January 2018 . The deepest point of the sea is 5 ft ( 1.5 m ) higher than the
lowest point of Death Valley . The sea is fed by the New , Whitewater , and Alamo rivers , as well as agricultural runoff , drainage systems , and creeks .
</item 1>

<item 2>
Salton Sea
Surface water temperature ( Sandy Beach ) Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec 57 ° F 14 ° C 57 ° F 14 ° C 64 ° F 18 ° C 72 ° F 22 ° C 77 ° F 25 ° C
82 ° F 28 ° C 90 ° F 32 ° C 91 ° F 33 ° C 86 ° F 30 ° C 81 ° F 27 ° C 72 ° F 22 ° C 61 ° F 17 ° C In popular culture ( edit ) An abandoned boat stuck in the
ground , close to the west coast marina of the Salton Sea Films The documentary Plagues & Pleasures on the Salton Sea ( which premiered at the 2004 Slamdance
Film Festival and , with the addition of John Waters as narrator , premiered at the Provincetown International Film Festival in Massachusetts and opened in
select theaters on February 24 , 2006 ) , covers the first 100 years of the Salton Sea , along with the environmental issues and offbeat residents of the region
.
</item 2>

<item 3>
Salton Sea
The Salton Sea is also a major resting stop on the Pacific Flyway . On 18 November 2006 , a Ross 's gull , a high Arctic bird , was sighted and photographed
there . Increasing salinity ( edit ) Dead fish on the western shore of Salton City The lack of an outflow means the Salton Sea does not have a natural
stabilization system ; it is very dynamic . Fluctuations in the water level caused by variations in agricultural runoff , the ancient salt deposits in the lake
bed , and the relatively high salinity of the inflow feeding the sea are all causing ever - increasing salinity . The body was initially a freshwater lake , but
by the 1960s its rising salinity had begun to jeopardize some of the species in it . With a salinity now exceeding 5.0 % w / v ( saltier than seawater ) , most
species of fish can no longer survive there . A freshwater fish notable for its ability to withstand the rising salinity of the Salton Sea , the desert pupfish
, can survive salinities ranging from 0.0 % to 7.0 % .
</item 3>

<item 4>
Salton Sea
Fertilizer runoffs have resulted in eutrophication , with large algal blooms and elevated bacterial levels . By 2014 , large swaths of lake bed were exposed and
salt levels drastically increased due to mandated water transfers to metropolitan areas along the coast and other factors , limiting the water inflow . Besides
the resulting fish kills , the shrinking lake interrupts the bird migration , causes dust clouds , and impacts local tourism negatively . Remediation efforts (
edit ) Past efforts and proposals for a Sea level canal ( edit ) Map of New River and Alamo River courses Alternatives for `` saving '' the Salton Sea have been
evaluated since 1955 . The New River passes from Mexicali , Baja California , to the Imperial Valley , and on to the Salton Sea . Much of the current interest
in the sea was sparked in the 1990s by Congressman Sonny Bono . His widow , Mary Bono Mack , elected to fill his seat , has continued to be interested in the
Salton Sea , as has Representative Jerry Lewis of Redlands . In 1998 , the Sonny Bono Salton Sea Restoration Project was named for the politician .
</item 4>

<item 5>
Salton Sea
Many of the species of fish that lived in the sea have been killed off by the combination of pollutants , salt levels , and algal blooms . Dead fish have been
known to wash up in mass quantities on the beaches . The smell of the lake , combined with the stench of the decaying fish , also contributed to the decline of
the tourist industry around the Salton Sea . Many people now visit the Salton Sea and the surrounding settlements to explore the abandoned structures . The town
of Niland is 1.5 miles ( 3 km ) southeast of the sea , with a population of 1,006 . Evidence of geothermal activity is also visible . Mudpots and mud volcanoes
are found on the eastern side of the Salton Sea . A number of geothermal electricity generation plants are located along the southeastern shore of the Salton
Sea in Imperial County . The US Geological Survey describes the smell as `` objectionable '' , `` noxious '' , `` unique '' , and `` pervasive '' .
</item 5>

<item 6>
Salton Sea
San Diego County Water Authority . Retrieved 2014 - 10 - 14 . Jump up ^ See http://www.usbr.gov/lc/region/programs/SaltonSea_2004-2014.xlsx , linked at
https://www2.usgs.gov/saltonsea/monitoring.html Jump up ^ Khaled M. Bali ( 5 October 2013 ) . `` Salton Sea and Salinity '' . UC Davis , Cooperative Extension
Imperial County . Retrieved 2015 - 06 - 29 . The salinity of the Pacific Ocean is approximately 35,000 mg / l ( 1 ppm = 1 mg / l ) . Colorado River water
salinity is about 650 -- 700 mg / L. Salton Sea salinity is about 44,000 mg / L , that is approximately 4.4 % salt . The amount of salts that is deposited in
the Imperial Valley agricultural land with irrigation water is approximately four million tons of salts annually . Jump up ^ Alles , DL ( 2007 - 08 - 06 ) . ``
Geology of the Salton Trough '' ( PDF ) .
</item 6>

<item 7>
Salton Sea
Salton Sea - wikipedia Salton Sea Jump to : navigation , search For other uses , see Salton Sea ( disambiguation ) . Salton Sea Location Colorado Desert
Imperial and Riverside Counties , California , U.S. Coordinates 33 ° 18 ′ N 115 ° 48 ′ W ﻿ / ﻿ 33.3 ° N 115.8 ° W ﻿ / 33.3 ; - 115.8 Coordinates : 33 ° 18 ′ N
115 ° 48 ′ W ﻿ / ﻿ 33.3 ° N 115.8 ° W ﻿ / 33.3 ; - 115.8 Type Endorheic rift lake Primary inflows Alamo River New River Whitewater River Primary outflows None
Catchment area 8,360 square miles ( 21,700 km ) Basin countries United States , Mexico Surface area 343 sq mi ( 889 km ) Max .
</item 7>

<item 8>
Salton Sea
Contra Costa Del Norte El Dorado Fresno Glenn Humboldt Imperial Inyo Kern Kings Lake Lassen Los Angeles Madera Marin Mariposa Mendocino Merced Modoc Mono
Monterey Napa Nevada Orange Placer Plumas Riverside Sacramento San Benito San Bernardino San Diego San Francisco San Joaquin San Luis Obispo San Mateo Santa
Barbara Santa Clara Santa Cruz Shasta Sierra Siskiyou Solano Sonoma Stanislaus Sutter Tehama Trinity Tulare Tuolumne Ventura Yolo Yuba Most populous cities Los
Angeles San Diego San Jose San Francisco Fresno Sacramento Long Beach Oakland Bakersfield Anaheim Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Salton_Sea&oldid=836852484 '' Categories : Salton Sea Endorheic lakes of California Lakes of Imperial County ,
California Lakes of Riverside County , California Lakes of the Great Basin Saline lakes Shrunken lakes Coachella Valley Imperial Valley Disasters in California
Engineering failures Environment of California Environmental disasters in the United States Geography of the Colorado Desert Important Bird Areas of California
Natural history of Imperial County , California Tourist attractions in Imperial County ,
</item 8>

<item 9>
Salton Sea
Originally , it was feared that the waterfall would recede upstream to the true main path of the Colorado , becoming up to 100 to 300 feet ( 30 to 91 m ) high ,
at which point it would be practically impossible to fix the problem . Salton Sea As the basin filled , the town of Salton , a Southern Pacific Railroad siding
, and Torres - Martinez Native American land were submerged . The sudden influx of water and the lack of any drainage from the basin resulted in the formation
of the Salton Sea . The U.S. Navy conducted a preliminary inspection of the Salton Sea in January 1940 , and the Salton Sea Test Base ( SSTB , run by Sandia
Labs ) was initially commissioned as the Naval Auxiliary Air Station , Salton Sea , in October 1942 . The SSTB , just to the south - east of Salton City ,
originally functioned as an operational and training base for seaplanes .
</item 9>

<item 10>
Salton Sea
Ecology ( edit ) See also : Sonny Bono Salton Sea National Wildlife Refuge § Wildlife Fish population ( edit ) Due to the high salinity , very few fish species
can tolerate living in the Salton Sea . Introduced tilapia are the main fish that can tolerate the high salinity levels and pollution . Other freshwater fish
species live in the rivers and canals that feed the Salton Sea , including threadfin shad , carp , red shiner , channel catfish , white catfish , largemouth
bass , mosquitofish , sailfin molly , and the vulnerable desert pupfish . Avian population ( edit ) The Salton Sea has been termed a `` crown jewel of avian
biodiversity '' by Dr. Milt Friend of the Salton Sea Science Office . Over 400 species have been documented at the Salton Sea . The most diverse and probably
the most significant populations of bird life in the continental United States are hosted , rivaled only by Big Bend National Park in Texas . It supports 30 %
of the remaining population of the American white pelican . The Salton Sea is also a major resting stop on the Pacific Flyway .
</item 10>
=== END ITEM id=6731f7b474590e90 ===

=== ITEM id=28a5662a4591cf27 ===
Gold sentences and facts:
Sentence 0: For thousands of years , seawater pearls were retrieved by divers in the Indian Ocean in areas such as the Persian Gulf , the Red Sea and the Gulf of Mannar . Evidence also suggest a prehistoric origin to pearl diving in these regions . Starting in the Han Dynasty ( 206 BC -- 220 AD ) , the Chinese hunted extensively for seawater pearls in the South China Sea . In the 14th - century Arabian Sea , the traveller Ibn Battuta provided the earliest known description of pearl diving by means of attaching a cord to the diver 's waist .
  Fact 0.0: For thousands of years, seawater pearls were retrieved by divers in the Indian Ocean.
  Fact 0.1: Seawater pearls were retrieved by divers in the Persian Gulf.
  Fact 0.2: Seawater pearls were retrieved by divers in the Red Sea.
  Fact 0.3: Seawater pearls were retrieved by divers in the Gulf of Mannar.
  Fact 0.4: Evidence suggests a prehistoric origin to pearl diving in the Persian Gulf, the Red Sea and the Gulf of Mannar.
  Fact 0.5: Starting in the Han Dynasty (206 BC to 220 AD), the Chinese hunted extensively for seawater pearls in the South China Sea.
  Fact 0.6: In the 14th-century Arabian Sea, the traveller Ibn Battuta provided the earliest known description of pearl diving.
  Fact 0.7: Ibn Battuta's description of pearl diving involved attaching a cord to the diver's waist.

Retrieved items:
<item 1>
Pearl
Journal of the Marine Biological Association of India . 15 : 1 -- 19 . Archived ( PDF ) from the original on July 18 , 2011 . Jump up ^ Arnold Wright ( 1999 ) .
Twentieth century impressions of Ceylon : its history , people , commerce , industries , and resources . p. 227 . ISBN 81 - 206 - 1335 - X . Archived from the
original on January 18 , 2017 . Jump up ^ James Hornell ( 2009 ) . The Indian Pearl Fisheries of the Gulf of Manar and Palk Bay . BiblioBazaar . p. 6 . ISBN 1 -
110 - 87096 - 5 . Archived from the original on January 18 , 2017 . Jump up ^ De Silva , K.M. ( 1995 ) . Volume 2 of History of Ceylon , History of Ceylon :
History of Sri Lanka . Peradeniya : Ceylon University Press . p. 56 . ISBN 955 - 589 - 004 - 8 .
</item 1>

<item 2>
Pearl
While white , and more recently black , saltwater pearls are by far the most popular , other color tints can be found on pearls from the oceans . Pink , blue ,
champagne , green , black and even purple saltwater pearls can be encountered , but to collect enough of these rare colors to form a complete string of the same
size and same shade can take years . Religious references ( edit ) This section needs additional citations for verification . Please help improve this article
by adding citations to reliable sources . Unsourced material may be challenged and removed . ( March 2018 ) ( Learn how and when to remove this template message
) Hindu scriptures ( edit ) The Hindu tradition describes the sacred Nine Pearls which were first documented in the Garuda Purana , one of the books of the
Hindu mythology . Ayurveda contains references to pearl powder as a stimulant of digestion and to treat mental ailments . According to Marco Polo , the kings of
Malabar wore a necklace of 104 rubies and pearls which was given from one generation of kings to the next . The reason was that every king had to say 104
prayers every morning and every evening .
</item 2>

<item 3>
Pearl
In 2010 , China overtook Japan in akoya pearl production . Japan has all but ceased its production of akoya pearls smaller than 8 mm . Japan maintains its
status as a pearl processing center , however , and imports the majority of Chinese akoya pearl production . These pearls are then processed ( often simply
matched and sorted ) , relabeled as product of Japan , and exported . In the past two decades , cultured pearls have been produced using larger oysters in the
south Pacific and Indian Ocean . The largest pearl oyster is the Pinctada maxima , which is roughly the size of a dinner plate . South Sea pearls are
characterized by their large size and warm luster . Sizes up to 14 mm in diameter are not uncommon . In 2013 , Indonesia Pearl supplied 43 percent of South Sea
Pearls international market . The other significant producers are Australia , Philippines , Myanmar and Malaysia . Freshwater pearl farming ( edit ) In 1914 ,
pearl farmers began growing cultured freshwater pearls using the pearl mussels native to Lake Biwa . This lake , the largest and most ancient in Japan , lies
near the city of Kyoto .
</item 3>

<item 4>
Pearl
It also records that eight varieties of pearls accompanied Prince Vijaya 's embassy to the Pandyan king as well as king Devanampiya Tissa 's embassy to Emperor
Ashoka . Pliny the Elder ( 23 -- 79AD ) praised the pearl fishery of the Gulf as most productive in the world . For thousands of years , seawater pearls were
retrieved by divers in the Indian Ocean in areas such as the Persian Gulf , the Red Sea and the Gulf of Mannar . Evidence also suggest a prehistoric origin to
pearl diving in these regions . Starting in the Han Dynasty ( 206 BC -- 220 AD ) , the Chinese hunted extensively for seawater pearls in the South China Sea .
In the 14th - century Arabian Sea , the traveller Ibn Battuta provided the earliest known description of pearl diving by means of attaching a cord to the diver
's waist . Catching of pearls , Bern Physiologus ( 9th century ) When Spanish conquistadors arrived in the Western Hemisphere , they discovered that around the
islands of Cubagua and Margarita , some 200 km north of the Venezuelan coast , was an extensive pearl bed ( a bed of pearl oysters ) .
</item 4>

<item 5>
Pearl
1 British Isles 7.2 Pearl farming 8 Timeline of pearl production 9 Freshwater pearl farming 10 Momme weight 11 In jewelry 11.1 Gallery 11.2 Shapes 11.3 Lengths
of pearl necklaces 11.4 Colors 12 Religious references 12.1 Hindu scriptures 12.2 Hebrew scriptures 12.3 New Testament scriptures 12.4 Islamic scriptures 12.5
Additional references 13 See also 14 References 15 External links Etymology ( edit ) The English word pearl comes from the French perle , originally from the
Latin perna meaning leg , after the ham - or mutton leg - shaped bivalve . Definition ( edit ) A black pearl and a shell of the black - lipped pearl oyster .
The iridescent colors originate from nacre layers . All shelled mollusks can , by natural processes , produce some kind of `` pearl '' when an irritating
microscopic object becomes trapped within its mantle folds , but the great majority of these `` pearls '' are not valued as gemstones .
</item 5>

<item 6>
Pearl
Prized for their exquisitely beautiful ' orient ' or lustre , South Sea pearls are now farmed in various parts of the world where the Pinctada maxima oysters
can be found , with the finest South Sea pearls being produced by Paspaley along the remote coastline of North - Western Australia . White and silver colored
South Sea pearls tend to come from the Broome area of Australia , while golden colored ones are more prevalent in the Philippines and Indonesia . A farm in the
Gulf of California , Mexico , is culturing pearls from the black lipped Pinctada mazatlanica oysters and the rainbow lipped Pteria sterna oysters . Also called
Concha Nácar , the pearls from these rainbow lipped oysters fluoresce red under ultraviolet light . From other species ( edit ) A shell of the Indian volute ,
Melo melo , surrounded by a number of pearls from this species Conch pearl pendant Biologically speaking , under the right set of circumstances , almost any
shelled mollusk can produce some kind of pearl . However , most of these molluskan pearls have no luster or iridescence .
</item 6>

<item 7>
Pearl
They are , for the most part , freshwater pearls from mussels . Pearling was banned in the U.K. in 1998 due to the endangered status of river mussels .
Discovery and publicity about the sale for a substantial sum of the Abernethy pearl in the River Tay had resulted in heavy exploitation of mussel colonies
during the 1970s and 80s by weekend warriors . When it was permitted it was carried on mainly by Scottish Travellers who found pearls varied from river to river
with the River Oykel in the Highlands being noted for the finest rose - pink pearls . There are two firms in Scotland that are licensed to sell pre-1998
freshwater pearls . Pearl farming ( edit ) A pearl being extracted from an akoya pearl oyster . See also : Oyster farming Today , the cultured pearls on the
market can be divided into two categories . The first category covers the beaded cultured pearls , including Akoya , South Sea and Tahiti . These pearls are
gonad grown , and usually one pearl is grown at a time . This limits the number of pearls at a harvest period .
</item 7>

<item 8>
Pearl
Jump up ^ Senaveratna , John M ( 1930 ) . The story of the Sinhalese : From the most ancient times up to the end of `` The Mahavasna '' , or great dynasty ;
Vijaya to Maha Sena ( B.C. 543 to A.D. 302 ) . W.M.A. Wahid . pp. 16 -- 9 , 46,130,144 -- 8,161 -- 3,213 . ASIN B000874SS0 . ISBN 9788120612716 . Archived from
the original on December 20 , 2017 . Jump up ^ Kunz , George F. ; Stevenson , Charles ( 1908 ) . The book of the pearl . New York : The Century Co. p. 2 .
Archived from the original on May 14 , 2016 . Jump up ^ R. Raghu Prasad and P.V. Ramachandran Nair ( 1973 ) . `` India and the Indian Ocean Fisheries '' ( PDF )
. Journal of the Marine Biological Association of India .
</item 8>

<item 9>
Pearl
Freshwater and saltwater pearls ( edit ) Freshwater and saltwater pearls may sometimes look quite similar , but they come from different sources . Freshwater
pearls form in various species of freshwater mussels , family Unionidae , which live in lakes , rivers , ponds and other bodies of fresh water . These
freshwater pearl mussels occur not only in hotter climates , but also in colder more temperate areas such as Scotland ( where they are protected under law ) .
Most freshwater cultured pearls sold today come from China . Saltwater pearls grow within pearl oysters , family Pteriidae , which live in oceans . Saltwater
pearl oysters are usually cultivated in protected lagoons or volcanic atolls . Creation ( edit ) Diagram comparing a cross-section of a cultured pearl , upper ,
with a natural pearl , lower Pearls are formed inside the shell of certain mollusks as a defense mechanism against a potentially threatening irritant such as a
parasite inside the shell , or an attack from outside that injures the mantle tissue . The mollusk creates a pearl sac to seal off the irritation .
</item 9>

<item 10>
Pearl
Australia also has one of the world 's last remaining fleets of pearl diving ships . Australian pearl divers dive for south sea pearl oysters to be used in the
cultured south sea pearl industry . The catch of pearl oysters is similar to the numbers of oysters taken during the natural pearl days . Hence significant
numbers of natural pearls are still found in the Australian Indian Ocean waters from wild oysters . X-ray examination is required to positively verify natural
pearls found today . Types of cultured pearls ( edit ) A blister pearl , a half - sphere , formed flush against the shell of the pearl oyster . Keshi pearls ,
although they often occur by chance , are not considered natural . They are a byproduct of the culturing process , and hence do not happen without human
intervention . They are quite small , typically only a few millimeters . Keshi pearls are produced by many different types of marine mollusks and freshwater
mussels in China . Keshi pearls are actually a mistake in the cultured pearl seeding process . In seeding the cultured pearl , a piece of mantle muscle from a
sacrificed oyster is placed with a bead of mother of pearl within the oyster .
</item 10>
=== END ITEM id=28a5662a4591cf27 ===

=== ITEM id=0fe313ac8c4a2333 ===
Gold sentences and facts:
Sentence 0: Queen Elizabeth II is the sovereign , and her heir apparent is her eldest son , Charles , Prince of Wales . Next in line after him is Prince William , Duke of Cambridge , the Prince of Wales 's elder son . Third in line is Prince George , the eldest child of the Duke of Cambridge , followed by his sister , Princess Charlotte and younger brother , Prince Louis . Sixth in line is Prince Henry of Wales , the younger son of the Prince of Wales . Any of the first six in line marrying without the sovereign 's consent would be disqualified from succession .
  Fact 0.0: Queen Elizabeth II is the sovereign.
  Fact 0.1: Charles, Prince of Wales, is the heir apparent to the sovereign, Queen Elizabeth II.
  Fact 0.2: Charles, Prince of Wales, is the eldest son of Queen Elizabeth II.
  Fact 0.3: Prince William, Duke of Cambridge, is next in line for succession after Charles, Prince of Wales.
  Fact 0.4: Prince William is the elder son of the Prince of Wales.
  Fact 0.5: Prince George is third in line for succession.
  Fact 0.6: Prince George is the eldest child of the Duke of Cambridge.
  Fact 0.7: Princess Charlotte follows Prince George in the line of succession.
  Fact 0.8: Princess Charlotte is the sister of Prince George.
  Fact 0.9: Prince Louis is the younger brother of Prince George.
  Fact 0.10: Prince Henry of Wales is sixth in line for succession.
  Fact 0.11: Prince Henry of Wales is the younger son of the Prince of Wales.
  Fact 0.12: Any of the first six in line to the throne who marries without the sovereign's consent would be disqualified from succession.

Retrieved items:
<item 1>
Succession to the British throne
Edward died childless in 1972 . Edward 's abdication was `` a demise of the Crown '' ( in the words of the Act ) , and the Duke of York , his brother who was
then next in the line , immediately succeeded to the throne and to its `` rights , privileges , and dignities '' , taking the regnal name George VI . He in turn
was succeeded in 1952 by his own elder daughter , Elizabeth II . By that time the monarch of the United Kingdom no longer reigned in the greater part of Ireland
( which had become a republic in 1949 ) , but was the monarch of a number of independent sovereign states ( Commonwealth realms ) . Commonwealth realms ( edit )
Further information : Perth Agreement Elizabeth II at her coronation in 1953 , passing to the left of the Coronation Chair Each of the Commonwealth realms has
the same person as monarch and , to maintain that arrangement , they have agreed to continue the same line of succession ; some realms do so through domestic
succession laws , while others stipulate whoever is monarch of the United Kingdom will also be monarch of that realm .
</item 1>

<item 2>
Succession to the British throne
Treason ( edit ) Under the Treason Act 1702 and the Treason ( Ireland ) Act 1703 , it is treason to `` endeavour to deprive or hinder any person who shall be
the next in succession to the crown ... from succeeding ... to the imperial crown of this realm '' . Since the Crime and Disorder Act 1998 , the maximum penalty
has been life imprisonment . Accession ( edit ) The Accession Council normally meets in St James 's Palace to proclaim the new sovereign . In the Commonwealth
realms , upon the death of a sovereign , the heir apparent or heir presumptive succeeds to the throne immediately , with no need for confirmation or further
ceremony . Nevertheless , the Accession Council meets and decides upon the making of the accession proclamation , which by custom has for centuries been
ceremonially proclaimed in public places , in London , York , Edinburgh and other cities . The anniversary is observed throughout the sovereign 's reign as
Accession Day , including royal gun salutes in the United Kingdom . Formerly , a new sovereign proclaimed his or her own accession . But on the death of
Elizabeth I an Accession Council met to proclaim the accession of James I to the throne of England .
</item 2>

<item 3>
Succession to the British throne
3 : `` A person who ( when the person marries ) is one of the 6 persons next in the line of succession to the Crown must obtain the consent of Her Majesty
before marrying . '' Jump up ^ Most sources , including Whitaker 's Almanack , exclude her as a confirmed Catholic , but Debrett 's includes her in line . Jump
up ^ Albert and Leopold Windsor were listed on The Official Website of the British Monarchy until 2015 and in the 2013 edition of Whitaker 's Almanack as
following Estella Taylor ( b 2004 ) and eligible to succeed ; MSN News , Debrett 's and Whitaker 's Almanack 2015 lists them after Lady Amelia Windsor and
before Lady Helen Taylor . They were baptised as Catholics , and are not listed in line in editions of Whitaker 's earlier than 2012 . Jump up ^ Neither the
Legitimacy Act 1926 nor the Legitimacy Act 1959 changed the Law of Succession . Jump up ^ Per the His Majesty 's Declaration of Abdication Act 1936 .
</item 3>

<item 4>
Succession to the British throne
In February 1952 , on her accession , Elizabeth II was proclaimed as sovereign separately throughout her realms . In October 2011 , the heads of government of
all 16 realms agreed unanimously at a meeting held in Perth , Australia , to proposed changes to the royal succession laws that would see the order of
succession for people born after 28 October 2011 governed by absolute primogeniture -- wherein succession passes to an individual 's children according to birth
order , regardless of gender -- instead of male - preference primogeniture . They also agreed to lift a ban on those who marry Roman Catholics . The ban on
Catholics themselves was retained to ensure that the monarch would be in communion with the Church of England . The changes came into effect across the
Commonwealth realms on 26 March 2015 , after legislation was made according to each realm 's constitution . Following the changes coming into effect , the
positions of the first 27 in line remained unchanged , including Princess Anne and her children and grandchildren , until the birth of Princess Charlotte of
Cambridge on 2 May 2015 .
</item 4>

<item 5>
Succession to the British throne
When updated in January 2011 , the number was more than five thousand . The annotated list below covers the first part of this line of succession , being
limited to descendants of the sons of George V ( the present Queen 's grandfather ) . The order of the first seventeen numbered in the list is given on the
official website of the British Monarchy ; other list numbers and exclusions are explained by annotations ( Notes and sources below ) and footnotes . Persons
named in italics are unnumbered either because they are deceased or because sources report them to be excluded from the succession . King George V ( 1865 --
1936 ) King Edward VIII ( 1894 -- 1972 ) King George VI ( 1895 -- 1952 ) Queen Elizabeth II ( born 1926 ) ( 1 ) Charles , Prince of Wales ( b . 1948 ) ( 2 )
Prince William , Duke of Cambridge ( b . 1982 ) ( 3 ) Prince George of Cambridge ( b . 2013 ) ( 4 ) Princess Charlotte of Cambridge ( b . 2015 ) ( 5 ) Prince
Louis of Cambridge ( b .
</item 5>

<item 6>
Succession to the British throne
Succession to the British throne Legislation Succession Act 1533 Succession Act 1536 Succession Act 1543 Coronation Oath Act 1688 Bill of Rights 1689 Act of
Settlement 1701 Act of Union 1706 Accession Declaration Act 1910 Perth Agreement 2011 Succession to the Crown Act 2013 Regency Acts Orders of succession
Monarchies Bahrain Belgium Bhutan Brunei Denmark Japan Jordan Liechtenstein Luxembourg Malaysia Monaco Morocco Netherlands Norway Saudi Arabia Spain Swaziland
Sweden Thailand Tonga United Kingdom and other Commonwealth realms Presidencies Argentina Austria Brazil Colombia D.R. Congo Dominican Republic France Finland
Germany Iceland India Indonesia Ireland Italy South Korea Mexico Pakistan Peru Philippines Poland Portugal Romania Russia Sri Lanka Taiwan United States
Retrieved from `` https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Succession to the British throne
Succession to the British throne - wikipedia Succession to the British throne This is the latest accepted revision , reviewed on 27 April 2018 . Jump to :
navigation , search The Sovereign 's Throne in the House of Lords , from which the speech is delivered at State Openings of Parliament Succession to the British
throne is determined by descent , gender ( for people born before October 2011 ) , legitimacy , and religion . Under common law , the Crown is inherited by a
sovereign 's children or by a childless sovereign 's nearest collateral line . The Bill of Rights 1689 and the Act of Settlement 1701 , restrict succession to
the throne to the legitimate Protestant descendants of Sophia of Hanover that are in `` communion with the Church of England '' . Spouses of Roman Catholics
were disqualified from 1689 until the law was amended in 2015 . Protestant descendants of those excluded for being Roman Catholics are eligible . Queen
Elizabeth II is the sovereign , and her heir apparent is her eldest son , Charles , Prince of Wales . Next in line after him is Prince William , Duke of
Cambridge , the Prince of Wales 's elder son .
</item 7>

<item 8>
Succession to the British throne
After Mary II died in 1694 , her husband continued to reign alone until his own death in 1702 . The line of succession provided for by the Bill of Rights was
almost at an end ; William and Mary never had any children , and Princess Anne 's children had all died . Therefore , Parliament passed the Act of Settlement .
The Act maintained the provision of the Bill of Rights whereby William would be succeeded by Princess Anne and her descendants , and thereafter by his own
descendants from future marriages . The Act , however , declared that they would be followed by James I & VI 's granddaughter Sophia , Electress Dowager of
Hanover ( the daughter of James 's daughter Elizabeth Stuart ) , and her heirs . As under the Bill of Rights , non-Protestants and those who married Roman
Catholics were excluded . Electress Sophia of Hanover ( 1630 -- 1714 ) Upon William 's death , Anne became Queen of England , Scotland and Ireland . Because the
Parliament of England settled on Sophia as Anne 's heir - presumptive without consulting Scottish leaders , the Estates of Scotland retaliated by passing the
Scottish Act of Security .
</item 8>

<item 9>
Succession to the British throne
Third in line is Prince George , the eldest child of the Duke of Cambridge , followed by his sister , Princess Charlotte and younger brother , Prince Louis .
Sixth in line is Prince Henry of Wales , the younger son of the Prince of Wales . Any of the first six in line marrying without the sovereign 's consent would
be disqualified from succession . The first four individuals in the line of succession who are over 21 , and the sovereign 's consort , may be appointed
Counsellors of State . Counsellors of State perform some of the sovereign 's duties in the United Kingdom while he or she is out of the country or temporarily
incapacitated . Otherwise , individuals in the line of succession need not have specific legal or official roles . The United Kingdom is one of the 16
Commonwealth realms . Each of those countries has the same person as monarch and the same order of succession .
</item 9>

<item 10>
Succession to the British throne
1 Stuarts 2.3. 2 Hanoverians and Windsors 2.4 Commonwealth realms 3 Current rules 3.1 Marriages 3.2 Religion 3.3 Treason 4 Accession 5 See also 6 Notes 7
References 8 Sources 9 External links Current line of succession ( edit ) First six in line from 23 April 2018 1 . The Prince of Wales 2 . The Duke of Cambridge
3 . Prince George of Cambridge 4 . Princess Charlotte of Cambridge 5 . Prince Louis of Cambridge 6 . Prince Henry of Wales No official , complete version of the
line of succession is currently maintained . The exact number , in remoter collateral lines , of the persons who would be eligible is uncertain . In 2001
genealogist William Addams Reitwiesner compiled a list of 4,973 living descendants of the Electress Sophia in order of succession , but did so disregarding
Roman Catholic status . When updated in January 2011 , the number was more than five thousand .
</item 10>
=== END ITEM id=0fe313ac8c4a2333 ===

=== ITEM id=dc45f4cd64c0b7b9 ===
Gold sentences and facts:
Sentence 0: Wizards of Waverly Place Season 4 episodes No . overall No. in season Title Directed by Written by Original air date Prod . code U.S. viewers ( millions ) 80 `` Alex Tells the World '' Victor Gonzalez Gigi McCreery & Perry Rein November 12 , 2010 ( 2010 - 11 - 12 ) 401 4.1 The Russos and Mason finally make it back home , but their wizard powers are still gone . While the family wants to stay quiet , Alex believes that they should take action , so she reveals that they are wizards to reporters and tells them that a group of wizards are being held captive by the government . As she urges them to spread the word , Professor Crumbs appears and tells her that everything that had happened was a test that she failed miserably and orders her to Wizard Court for exposing wizardry to the world . Justin is also ordered to attend because he exposed magic to Agent Lamwood , who was a creation from Professor Crumbs ' mind . The judges rule Alex and Justin guilty , and Professor Crumbs sentences them both to be demoted to level one in the Family Wizard Competition , making them behind Max by two levels . Meanwhile , Max sets up traps in the Sub Station for the `` government '' , but ends up falling for one of them . Special guest star : Gregg Sulkin as Mason Guest star : Ian Abercrombie as Professor Crumbs 81 `` Alex Gives Up '' Victor Gonzalez Todd J. Greenwald November 27 , 2010 ( 2010 - 11 - 27 ) 402 N / A Alex decides to give up the wizard competition to spend more time with Mason and thinks that it does not matter since Max is all the way in the lead , but Chancellor Tootietootie tells Alex that non-wizards and werewolves can not date , because the werewolf always eats the human . Max 's picture on the front of Future Wizards magazine attracts the daughter of a family of Cucuys ( mythical Latino monster ) and they invite him to a dinner party on their yacht . Alex comes up with a scheme to invite Chancellor Tootietootie to the Cucuy party in hopes of getting him to appeal the law against werewolves and mortals dating . Lisa Cucuy finds out that Mason is a werewolf and decides that she wants him instead of Max . Alex gets jealous when Lisa flirts with Mason , so Alex says embarrassing things about Mason to everyone on the yacht so that he will be mad enough to turn into a werewolf . Carlos and Julie Cucuy jump off the boat as Cucuys are scared of werewolves . Jerry jumps off after them . Alex tells Lisa that she can get off the boat the easy way or her way . Once Lisa jumps off the boat , Chancellor Tootietootie says that their case will not be appealed as by Mason turning into a werewolf proved his point that they are dangerous . In the end , Alex and Mason have to break up to be friends and Harper finishes writing her first novel based on the werewolf - wizard relationship . Jerry tries to get Harper to help him out of the water but accidentally grabs her just finished novel , which causes him to fall back into the water . Meanwhile , Justin tutors a class of delinquent wizards in an attempt to gain a level to re-enter the Wizard Competition . Special guest star : Gregg Sulkin as Mason Guest stars : Andy Kindler as Chancellor Tootietootie , Frank Pacheco as Felix , David Barrera as Carlos Cucuy , Roxana Brusso as Julie Cucuy , Samantha Boscarino as Lisa Cucuy , Kari Wahlgren as Helen 82 `` Lucky Charmed '' Victor Gonzalez Justin Varava December 10 , 2010 ( 2010 - 12 - 10 ) 403 3.0 Harper tries teaching Alex how to live without magic , but the strain of taking responsibility for her actions instead of using magic causes a disagreement between the two best friends when Alex gets in big trouble for crashing Mr. Laritate 's car . As punishment , Alex has to clean the school 's garbage cans . Mr. Laritate forgives Alex after seeing how the crash influenced her , and Alex realizes that she will be okay without magic . Wanting to thank Harper , Alex uses Mr. Laritate 's lucky life - saving hula girl figurine , Leilani , for luck , as Mr. Laritate does . During Harper 's driving test , Alex turns into the hula girl and guides Harper through the test , knowing that she might fail . In the end , Harper passes her test . Meanwhile , Jerry wants to pass on the wizard family robe to Max , but Justin has a hard time letting go of it . Guest stars : Bill Chott as Mr. Laritate , Julio Oscar Mechoso as DMV Instructor Absent : Maria Canals Barrera as Theresa Russo 83 `` Journey to the Center of Mason '' Robbie Countryman Peter Dirksen December 17 , 2010 ( 2010 - 12 - 17 ) 404 N / A After she finds out that she will lose her powers , Alex can no longer date Mason , so they decide to try being best friends instead . After watching a film at the theaters , Harper , Alex , and Mason go home , Dean , who was Alex 's Ex-Boyfriend comes and asks Alex to get back together with him . Mason gets jealous . When Dean brings a gift for Alex , Mason then asks what Alex 's relationship is with Dean , and Max tells him that Dean and Alex used to date , making Mason more jealous . Mason then turns into a werewolf and eats Dean . Alex , Justin , and Max come up with a solution to turn Justin 's toy Captain Jim Bob submarine big and go in it . They then shrink it and go into Mason 's mouth . While there , Alex , Max , and Justin see Mason 's thoughts , which are all of Alex . They get Dean out , and Alex , when Harper , Max , and Justin leave , Alex tells Dean that he was knocked out and the doctor 's solution was for them all to dress up as cows . Dean then gives her his offer once again : he wants to get back with her . Alex tells him ' no ' , and then gets back together with Mason , with deciding to get back to the Wizard Competition . Special guest star : Gregg Sulkin as Mason Guest stars : Daniel Samonas as Dean , Frank Pacheco as Felix , Shoshana Bush as Tyler 84 5 `` Three Maxes and a Little Lady '' Victor Gonzalez Richard Goodman January 7 , 2011 ( 2011 - 01 - 07 ) 405 4.0 Alex joins Justin 's wizard class to try to move up in the competition . Max joins a sophisticated wizard society . They suggest he move the wizard competition up and he moves it to a few days away . Max quits the society because he finds out that they are taking advantage of him . Alex and Justin transform into Max in an attempt to change the date of the family wizard competition back to its original date and succeed . In the end of the episode they accidentally turn Max indefinitely into a little girl . Guest stars : Bailee Madison as Maxine , Frank Pacheco as Felix , Kari Wahlgren as Helen , Shane Harper as Fidel Note : Bailee Madison begins her recurring role as Maxine , the female Max in this episode . Jake T. Austin was absent in four other episodes and returns at the end of the tenth episode `` Back to Max '' . 85 6 `` Daddy 's Little Girl '' Robbie Countryman Gigi McCreery & Perry Rein January 21 , 2011 ( 2011 - 01 - 21 ) 406 3.4 With Max still a little girl , their parents pay more attention to Max ( Maxine ) , and Alex becomes jealous . Justin takes Maxine to her ( Max 's ) karate class and Maxine kicks Justin 's butt . After the class , Alex decides to make Jerry 's favorite sandwich in order to win his love back , but Maxine comes along with Justin , announcing that she beat Justin in karate , and Jerry immediately falls for her cuteness again . More jealous than ever , Alex comes back with Justin to Maxine 's karate class with their parents , having a slight idea of knowing how to change Maxine back to Max . They cast spells on Maxine , Justin using the spell he used on Max that turned him into Maxine , and Alex using the reverse spell . Unfortunately , it only makes Maxine cuter , which makes her furious . Instead of going against the person she was supposed to , she demands that she kicks either Alex or Justin 's butt . Justin goes against her , but seeing that he would lose , Alex casts a spell on Justin to make him a karate master . Too fast for Maxine , Justin beats her , and Maxine is hurt ( physically ) . Jerry and Theresa rush to her side , caring for Maxine more than they care that Justin beat Maxine in karate . Justin leaves , and Alex leaves after him , heartbroken . Back at home , Jerry tells Alex that he fell for Maxine because she reminded him of Alex when she was little and they bond again . Meanwhile , Zeke thinks the Russos hate him , but really it was just Harper 's alibi in order to keep him away from the Sub Station so as not to make him suspicious about Max 's disappearance and Maxine 's appearance . Guest stars : Bailee Madison as Maxine , Dan Benson as Zeke , Lanny Joon as Sensei Absent : Jake T. Austin as Max Russo 86 7 `` Everything 's Rosie for Justin '' Victor Gonzalez Justin Varava February 4 , 2011 ( 2011 - 02 - 04 ) 407 3.9 Justin and his delinquent class have a wizard wand dance assessment to do which will get them back in Wiztech . A new student named Rosie joins , whom Justin falls madly in love with to later find out that she is an Angel . Meanwhile , Jerry , Teresa , and Harper try to think of ideas of how to get more people to come to their restaurant . Harper makes cards which customers get hole punched each time they buy a sandwich ; when you get 9 hole punches , they get a free sandwich . Contrary to the original purpose , one man punches his own card , demands a free sandwich , and then brings dozens of people into the restaurant , who also have pre-punched cards . Maxine uses her charming cuteness to convince the customers to not exploit her parents . Meanwhile , Justin kicks Rosie out of his class so that it does not fail the wizard dance assessment . Sympathizing , Alex gets Rosie to rejoin the class and puts a copycat spell on her so that Rosie can mimic Alex 's move and pass the class . The assessment ends in disaster when a squirrel - frog attaches itself to Alex 's foot , and Rosie reveals herself to be an angel . The class fails , but Justin starts dating Rosie . Guest stars : Bailee Madison as Maxine , Frank Pacheco as Felix , Leven Rambin as Rosie , Diane Delano as Penny , Chris Coppola as Customer , Cameron Sanders as Nelvis Absent : Jake T. Austin as Max Russo 87 8 `` Dancing with Angels '' Victor Gonzalez Richard Goodman February 11 , 2011 ( 2011 - 02 - 11 ) 408 4.5 Justin and Rosie start dating but when their date gets ruined , they want a second date . Alex suggests an Angel Club in Los Angeles , but Rosie says that only Angels are allowed . Maxine will not tell their parents as long as Alex brings her back an Ozzy Osbourne stone . After Justin , Harper and Alex sneak in along with Rosie , Zeedrik asks them if they are angels and they say they are . Back at the sub shop , Theresa and Jerry question Maxine to find out where Justin , Alex and Harper are , and Maxine does not say , so they force her to join a Pageant in order for her to tell them where they are . Things get bad at the angel club when Alex and Harper can not fly , until Justin secretly pulls out his wand and gives Harper flying powers . Zeedrik still questions them by giving them a test to sing while playing the harp . Though Alex , Justin , and Harper fail to do so , they are accepted as `` angels '' until the key - chain of Ozzy Osbourne 's star reverts to normal . Zeedrik believes they are really angels of darkness , causing everyone in the club to run off , even Rosie . Rosie comes back at the end of the episode to comfort Justin , but when they both look over the city , Rosie 's wings are shown to be black , the color of an angel of darkness ' wings , but Justin does not see because he is looking forward . Meanwhile , Maxine ends up winning the beauty pageant , making Jerry and Theresa celebrate . Guest stars : Bailee Madison as Maxine , Leven Rambin as Rosie , Ransford Doherty as Zeedrik Absent : Jake T. Austin as Max Russo 88 9 `` Wizards vs. Angels '' Victor Gonzalez Vince Cheung & Ben Montanio February 18 , 2011 ( 2011 - 02 - 18 ) 409 -- 410 5.1 Rosie is an Angel of Darkness , and has been tricking Justin all along to make him evil . Alex finds out , with the help of Tina ( a guardian angel in training trying to earn her wings ) that Rosie is an Angel of Darkness . Justin crosses over to the dark side with Rosie , steals the `` moral compass '' , and gives it to Gorog , so Gorog can turn the compass from good to bad , so that the world will be covered in darkness and every human being will be corrupted . Rosie tries to get Justin to run away and save him , because Gorog wants to destroy him , but Justin refuses to leave , stating that he is no longer a wizard but an Angel of Darkness and breaks his wand in halves . Alex tries to convince Justin that he is a wizard , a good wizard and not an Angel of Darkness . Meanwhile , Maxine is enrolled in Tribeca Prep , since she is participating in a `` cousin exchange program . '' After her first day of school , Maxine laments that she has no friends , which leads Harper to insist Maxine have a slumber party ( which is really what Harper wants because she has never had a slumber party ) , that eventually becomes a disaster . Tina reveals that Rosie was a good angel , and her teacher however , Gorog reached her before her transport arrived and corrupted her . Alex and Tina go to the Dark Realm to save Justin . Alex borrows a pair of wings from the guardian angels . Alex takes the Moral Compass and tries to turn it to good but Justin stops her and then Rosie used her good angel side and they fly up and when they land , Justin turns the compass to good . Tina and Alex take it and run . When Rosie and Justin kiss , Rosie turns into a good angel ( with white wings ) , and Justin loses the dark wings . Guest stars : Bill Chott as Mr. Laritate , Bailee Madison as Maxine , Leven Rambin as Rosie , John Rubinstein as Gorog , China McClain as Tina , Amy Tolsky as Florence Note : This is a one - hour special . Absent : Jake T. Austin as Max Russo 89 10 `` Back To Max '' Guy Distad Todd J. Greenwald March 11 , 2011 ( 2011 - 03 - 11 ) 411 3.7 Justin invites Professor Crumbs to check on his delinquent wizards class , to see how their wizard training is progressing . During the conversation , Maxine interrupts , causing Alex and Justin to panic since they do not know how to explain `` Maxine '' , and they try to find a way to turn Maxine back to Max . Meanwhile , Harper takes a role in the school 's `` Spirit of America '' play , only to be replaced by Maxine . Guest stars : Bill Chott as Mr. Laritate , Ian Abercrombie as Professor Crumbs , Bailee Madison as Maxine , Cameron Sanders as Nelvis , LJ Benet as Little Crumbs , Jackie Evancho as Choir Girl Note : Jake T. Austin returns as Max near the end of the episode . Bailee Madison marks her final appearance as Maxine in this episode . 90 11 `` Zeke Finds Out '' David DeLuise Peter Dirksen April 8 , 2011 ( 2011 - 04 - 08 ) 412 3.0 When Alex gets bored of Zeke 's failing magic tricks , she decides to use magic to make Zeke 's tricks come true . Everything goes downhill when Zeke tells Harper he believes he is a wizard and is now booking a magic act at the sub station for a kid 's birthday party . Since Justin and Harper are tired of Alex ruining things all the time , they decide to leave everything up to her . Harper , not wanting to let Zeke make a fool of himself , tells Alex to tell Zeke the truth about wizards , but she is reluctant to , as exposing wizardry to another mortal could get her kicked out of the wizard competition forever if anyone finds out . Zeke goes on with his belief that he is a `` wizard '' , but he finds out Harper is keeping a secret from him so he breaks up with her . Alex realizes what a mess she created and on the day of Zeke 's magic act , she makes Harper disappear before she can expose wizardry to Zeke , which freaks him out . The Russos really are in deep waters when Zeke is going to saw Max in half with a real saw for his next act , so Justin takes matters into his own hands and freezes everyone except for himself and Zeke . Justin tells Zeke that he is a wizard , not Zeke , and that he did everything . Still not believing him , he flashes Zeke into their lair and unveils the secret to him . They go back outside and tell Harper and Alex who are fighting , that Zeke now knows about the existence of wizards , but he promises he will never tell anyone . Zeke and Harper reconcile their relationship and Alex and Harper apologize and make up with each other . Guest stars : Dan Benson as Zeke , Maurice G. Smith as Big Mitch Absent : Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 91 12 `` Magic Unmasked '' David DeLuise Justin Varava May 13 , 2011 ( 2011 - 05 - 13 ) 413 2.7 When a former legendary Luchador `` Muy Macho '' visits the Sub Station , Alex finds out that she was the reason why he quit lucha libre ( wrestling ) , so she tries to make it up to him by arranging a lucha libre match with him and Jerry to restore his honor . Meanwhile , Zeke asks too many questions about wizardry , so Justin offers to grant him one magical wish if he promised he would stop questioning him about magic . Unfortunately , Zeke 's wish is to defeat Muy Macho in the match . Max pretends to like Talia 's hobbies , but does not enjoy them . Guest stars : Dan Benson as Zeke , Valente Rodriguez as Muy Macho , McKaley Miller as Talia 92 13 `` Meet the Werewolves '' Victor Gonzalez David Henrie June 17 , 2011 ( 2011 - 06 - 17 ) 415 4.0 When Alex asks Mason if she can meet his parents , Mason shows his fake parents and she finds out soon after . She gets upset at Mason and demands him to take her to his real parents . Mason takes her to the Autumn Moon Feast ( werewolf holiday ) to meet his parents . Soon after that Alex finds out he told his parents that she is a werewolf and is frustrated . She uses a spell to change herself and Harper into werewolves . She tries to be the worst werewolf possible in front of Mason 's parents during the feast . In the end Mason 's parents like her as a werewolf and she says to him that they do not know the real her . She begins to leave and he says that he truly loves Alex to his parents and that she is a wizard . She changes back to a wizard and Mason turns back to human form . After the parents find out they try to eat her and Mason and Alex make a near escape . Meanwhile Max makes his parents ' food taste like kid food in order to make it taste good . Afterwards Jerry and Theresa eat it , and as a result they act like little kids . Justin thinks he will have to fix it but Max comments on how he does not need help , as he will be the future family wizard . Max uses another spell on cookies to make them older , but they turn into teenagers . Finally , Justin uses a spell on a pizza to turn them back into their regular selves . Special guest star : Gregg Sulkin as Mason Guest stars : Harry Van Gorkum as Grant Greybeck , Robin Riker as Linda Greybeck , Marianne Muellerleile as Molly , Eric Zuckerman as Bill 93 14 `` Beast Tamer '' Victor Gonzalez Gigi McCreery & Perry Rein June 24 , 2011 ( 2011 - 06 - 24 ) 416 4.0 When Alex , Justin and Max win four tickets to the `` Beast Bowl '' , the tickets are delivered by Beast Tamer Chase Riprock , who begins flirting with Alex and invites the group for a pre-bowl tour . Alex invites Mason to go with them but he says he can not . During the tour , despite Chase 's warnings , Justin attempts to tame the beast so that Max can have his photo taken with the beast , but both are caught and banned from the event . Chase tries to kiss Alex but she rejects him and returns to the mortal world where she finds Mason coming out a hardware store . Mason is evasive about his plans , which involve making an anniversary present for Alex , so Alex returns to the Beast Bowl. Harper tells Mason that Chase likes Alex and urges him to surprise her with the gift at the Beast Bowl . Meanwhile , Justin and Max return to the Beast Bowl disguised as beast taming clowns where they allow the beast to escape and snatch Alex who throws the whip to Chase . Chase tames the beast who drops Alex and Mason is able to reveal his gift , which is a sculpture of himself and Alex. Alex and Mason leave together , with the sculpture . Special guest star : Gregg Sulkin as Mason Guest star : Nick Roux as Chase Absent : Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 94 15 `` Wizard of the Year '' Victor Gonzalez Richard Goodman July 8 , 2011 ( 2011 - 07 - 08 ) 417 3.6 Alex has been crowned Wizard of the Year after saving the world from dark angels , as well as being reinstated in the Wizard Competition , and a banquet honoring her achievement is hosted . With paparazzi in tow , Chase Riprock visits the lair to congratulate Alex . As he leaves , Keith Keith hounds them about being in a romantic relationship , to which Alex states she already has a boyfriend . Later , Mason and Alex watch his gossip segment , and to their horror , Keith Keith confirms that Chase and Alex are dating . For Alex winning Wizard of the Year , the Russo family must record a holographic video congratulating Alex . However , much to Justin 's displeasure , Justin 's video instead only contains him complaining about Alex . The night of the awards banquet , Mason stands Alex up , leaving her devastated . At the banquet , Chancellor TootieTootie tells Justin that his display of affection and pride for Alex may reinstate him into the competition , causing Justin to be frantic and do everything within his power to destroy the tape . Chase purposely invites himself to sit at the Russo 's table . However , Mason shows up late and sees Alex and Chase flirting together . A short battle ensues between Mason and Chase . Alex soon breaks up the fight , and is forced to choose either Mason or Chase . Appalled by both of their behaviors that evening , Alex chooses neither and breaks up with Mason . Special guest star : Gregg Sulkin as Mason Guest stars : Andy Kindler as Chancellor Tootietootie , Nick Roux as Chase , Justin Guarini as Keith Keith 95 16 `` Misfortune at the Beach '' Jody Margolin Hahn Vince Cheung & Ben Montanio July 24 , 2011 ( 2011 - 07 - 24 ) 420 3.5 The Russos head to the beach on an extremely hot day , and Harper hopes to enjoy reading her book while at the beach . The Russos come across Zelzar , a fortune - telling machine . Jerry warns the kids about Zelzar -- he is from the wizard world , and a wizard 's fortune really comes true . After Max and Justin receive good fortunes , Alex decides to get a fortune despite her father 's warning . However , her fortune says , `` Say good - bye to your life . '' Alex thinks nothing of it and proceeds to enjoy the day at the beach , but comes close to life - threatening situations multiple times . Alex , Justin , Max , and Harper make a compromise with Zelzar -- if they can extract him from the machine and let him enjoy a day at the beach , he will take back Alex 's fortune . Max replaces Zelzar , and Justin is forced to drive the beach - goers away from the machine ( as Max gives out silly fortunes about his new facial hair ) ; meanwhile , the girls help Zelzar fancy his day at the beach . In the end , Zelzar reveals that while he can not take away Alex 's fortune , he can give it to the next person . The next person happens to be a little girl , who gets Alex 's fortune . However , it is revealed that a man from Random Prize Giveaway shows up , saying : `` Say good - bye to your life because we 're randomly giving you one million dollars ! '' Guest stars : Michael Carbonaro as Zelzar , John J. York as Game Show Host 96 17 `` Wizards vs. Asteroid '' Victor Gonzalez Peter Dirksen August 19 , 2011 ( 2011 - 08 - 19 ) 418 3.6 When it is announced that a huge asteroid is hurtling towards the earth , the Russos decide to hide in the wizard world , but Alex , Justin , and Max change their minds and go into space to destroy the asteroid . Meanwhile , Alex finds out that she is not graduating high school , but when a small part of the asteroid hits the school , she gets a chance to graduate . Guest stars : Dan Benson as Zeke , Bill Chott as Mr. Laritate , Kristina Hayes as Reporter 97 18 `` Justin 's Back In '' Guy Distad Todd J. Greenwald August 26 , 2011 ( 2011 - 08 - 26 ) 419 3.2 Justin 's class of delinquent wizards has completed their studies , which means they are eligible to rejoin WizTech if they all pass their final evaluation . When they all fail , Alex suspects something is not right . The truth soon uncovered -- the class had actually passed , but the historian who checked the exams was revealed to be evil , and had lied about the exam scores . He tries to attack Justin 's class , so Justin must stop the historian , and save Alex. but one of the students in the class is able to stop him by pulling the most powerful wand in the wizard world out of a crystal ball and using it . This means that he is actually the descendant of a famous wizard who had previously been thought to have no living descendants . Justin 's class is revealed to have passed the final exam , so they are readmitted into WizTech and Justin is officially back in the family wizard competition . Meanwhile , Jerry and Theresa realize that they never saved any of Max 's childhood accomplishments , so they try to recreate them with the help of Harper . During the credits , it is revealed that Max had kept his favorite memories all along . Special guest star : Tim Conway as Cragmont Guest stars : Ian Abercrombie as Professor Crumbs , Frank Pacheco as Felix , Cameron Sanders as Nelvis 98 19 `` Alex the Puppetmaster '' Victor Gonzalez Gigi McCreery & Perry Rein September 16 , 2011 ( 2011 - 09 - 16 ) 421 3.0 Alex and Harper decide to do a puppet show to raise money for a new apartment and Zeke and Justin help them . But Alex forgets to write the lines for the show . Harper gets mad and decides to do a show of her own and makes her puppets act like the way Alex treats her . Alex makes Justin and Zeke puppets for her show and steals Harper 's crowd which makes Harper upset . Then Alex apologizes to Harper using the puppets . Meanwhile , Talia 's parents meet Theresa and Jerry for the first time , but it does not go well . Justin puts a spell on Talia 's parents to forget they ever meet Theresa and Jerry . They meet again , but this time they end up watching a 16 - hour play together . Special guest star : Joely Fisher as Meg Robinson Guest stars : Dan Benson as Zeke , James Urbaniak as Rob Robinson , McKaley Miller as Talia 99 20 `` My Two Harpers '' Victor Gonzalez Manny Basanese September 30 , 2011 ( 2011 - 09 - 30 ) 422 3.0 Harper and Zeke spend so much time together that Alex decides to make a Harper - clone . Justin comes with lots of different ideas to save money in the sub-shop and Max starts a fast - food restaurant in the wizard world . Guest star : Dan Benson as Zeke Absent : Maria Canals Barrera as Theresa Russo 100 21 `` Wizards of Apartment 13B '' Guy Distad Justin Varava October 7 , 2011 ( 2011 - 10 - 07 ) 423 3.5 Alex and Harper receive a flyer for a hotel . They meet the hotel 's owner , Dexter , who sells them Apartment 13B , which belongs to a floor for non-humans . They rent it , but Harper is forced to use a training wand that can only open things . Later at their apartment , they realize Mason lives on the floor . Mason will stop at nothing to win Alex back ; Alex tells him she does not want a boyfriend , but Mason is relentless . That night , Alex and Harper throw a party , where the whole floor comes , including Felix . Mason goes to the party and uses Harper 's wand to open Alex 's heart . However , she flirts with an ugly giant . Mason tries again , but makes Alex opens her heart to Felix . Meanwhile , Theresa and Jerry trick Justin into going to Alex 's apartment to drive him out of the house . When Justin arrives , he fixes Mason 's mistakes by resetting Harper 's wand and it is revealed that despite his actions , he misses Alex. Alex , freed from Mason 's spell , is angry at Mason , and Mason , deflated , leaves . Outside their door , Dexter turns into Gorog when no one is looking . He wants revenge . Max appears in the apartment , and when Harper and Alex kick him out , he encounters Gorog , who gives him a flyer for a free camp . Max accepts . Special guest star : Gregg Sulkin as Mason Guest stars : John Rubinstein as Gorog , Frank Pacheco as Felix , Fred Stoller as Dexter , Christopher Douglas Reed as Ogre Note : This is part 1 of a 4 - part arc and also the 100th episode . 101 22 `` Ghost Roommate '' David DeLuise Richard Goodman October 14 , 2011 ( 2011 - 10 - 14 ) 424 4.1 Alex and Harper have trouble paying the apartment bills , so Dexter recommends getting a roommate . Alex and Harper find a wealthy ghost named Lucy , who they let move in . Meanwhile , Justin orders a robot to help Zeke at the sub station , as Justin is busy studying for the wizard competition . The robot is lost in transit , but Dexter ( Gorog in disguise ) makes an evil robot to spy on Justin to pass information to Gorog . While Zeke and the robot are working , the robot asks him about Justin and the wizard competition , which confuses Zeke because no one had talked about it in front of the robot . Later that night , Lucy starts crying and explains that her boyfriend Donny had disappeared sixty years prior in a plane accident . Alex tries to set Lucy up with Justin , but he makes matters worse . Later , Lucy meets Mason and they go out on a long date , which makes Alex jealous . Meanwhile , the robot again asks Zeke about the wizard world and Zeke realizes that the robot might have a secret agenda . He tells Justin , who simply does not believe him and is ignorant of the situation because of his intensive studying . Because of her jealousy , Alex helps Lucy locate Donny , who had been stranded on an island in the Bermuda Triangle . They soon reconnect , but not after Alex realizes that she can not get back home , due to heavy magnetic currents interfering with her magic . Mason goes to save Alex after he finds out that she had left for the Bermuda Triangle . The robot attacks Justin and Zeke after they find him scanning the spell book . Mason and Alex manage to get off of the island using the Bermuda shorts given to them by Justin . The robot is shown giving Gorog the spell he needed that he got from the Russos . Special guest star : Gregg Sulkin as Mason Guest stars : Dan Benson as Zeke , Fred Stoller as Dexter , Josh Sussman as Hugh Normous , Linsey Godfrey as Lucy , Travis Caldwell as Donny , Michael Carbonaro as Robot Note : This is part 2 of a 4 - part arc . Absent : Jake T. Austin as Max Russo , Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 102 23 `` Get Along , Little Zombie '' Victor Gonzalez Peter Dirksen October 21 , 2011 ( 2011 - 10 - 21 ) 425 3.4 Alex and Harper are going to their apartment when they run into Mr. Laritate , who also lives in the hotel . Felix tells Justin he broke his wand and Justin informs him that it 's fake , because he saw batteries in it . He concludes that someone stole Felix 's wand . Dexter is showing a zombie to the hall for magical creatures while he introduces them to Alex and Harper . Mason , who will stubbornly not stop asking Alex on a date , meets them in the elevator . When Alex ignores him , Mason messes with the buttons so he can spend more time with her and they land on the second floor . Justin tells Jerry that Felix lost his wand and Jerry tells them to use the abracadoobler wand app and Felix realizes that his wand is around the area . Meanwhile , because Mason messed with the buttons , all the humans have discovered the mysterious thirteenth floor . Alex lies and tells the humans that they accidentally got into a secret open house floor for the people in the hotel . Mr. Laritate , after all the other humans left , discovers the thirteenth floor too and sees Alex , Harper , and Mason . Mr. Laritate sees the zombie , Alex explains this is a haunted house , but Mr. Laritate does not believe them . The zombie bites Mr. Laritate and turns him into a zombie . They take him to their apartment while Alex looks for a spell to turn Mr. Laritate human again . Mr. Laritate runs away and Alex , Harper , and Mason suspect he went to the substation . Professor Crumbs questions Felix , who reveals the 13th floor to them , and Crumbs denies the existence of a `` 13th floor '' , so they go to investigate . At Waverly Place they 're throwing a festival while Alex tells Jerry what happened . Alex and Mason , trying to sneak the zombie into the substation , are forced into dance . They then , after Harper causes a distraction , take Laritate to the substation and give him a medicine from the wizard world that turns Laritate human . He remembers nothing and goes outside to dance . Alex kisses him on the cheek , and Mason suggests they just be friends . Alex gets back together with Mason . Crumbs tells Felix the floor should not exist . Harper , Mason , and Alex return to the thirteenth floor . Justin tells them that the floor does not exist . Crumbs thinks that there is evil going on . Dexter then comes with dark angel wings and it is revealed he stole Felix 's his wand . Dexter tricks Felix into turning to the dark side in return for the wand . Felix casts a spell that traps them all on the floor . Back at Waverly Place , Mr. Laritate is dancing while Jerry is selling chili , when Mr. Laritate gets one and gets water to get `` the burn '' out of his mouth . Special guest star : Gregg Sulkin as Mason Guest stars : Bill Chott as Mr. Laritate , Ian Abercrombie as Professor Crumbs , Frank Pacheco as Felix , Fred Stoller as Dexter , Regan Burns as Everett , Jane Carr as Martha St. Claire Note : This is part 3 of a 4 - part arc . Absent : Jake T. Austin as Max Russo , Maria Canals Barrera as Theresa Russo 103 24 `` Wizards vs. Everything '' Victor Gonzalez Gigi McCreery & Perry Rein October 28 , 2011 ( 2011 - 10 - 28 ) 426 3.4 The wizards of 13th floor come out and Alex explains Dexter 's evil . Dexter reveals himself as Gorog , and that he is trying to rebuild his army of darkness . Felix uses a spell that makes all the wizards from the 13th floor go with them , so they can make them all evil , all except Alex , Justin , and Harper , who are left behind . A trashcan tells them to go through it , however Alex ends up in the room again . Justin realizes only mortals can go through it . Alex convinces Harper into going through it , Harper then arrives in the wizard lair . Gorog takes the kidnapped wizards to his evil lair , where he plans to make them evil . He tricks Hugh Normous into becoming one of the evil wizards in his army . Professor Crumbs fakes becoming evil when Gorog plans on making him evil . Harper contacts Max and tells him about Gorog . Max arrives and suggests they look for spells in the wizard books . Gorog makes the wizard army to dig a hole to the atmosphere to the wizard world , so they can take over it . Gorog also tries to trick Mason into being one of the members in his army , but Mason refuses . Gorog tricks him into joining , however . Gorog then makes Felix send a black whole to suck Alex and Justin in it . Gorog gives the kidnapped wizards their wands so they can take over the wizard world , but Crumbs refuses to let them . Felix then gets rid of Crumbs on Gorog 's order . The black hole nearly sucks Alex and Justin when Crumbs arrives at the wizard lair . Max suggests they put a black hole in the wizard lair . Max goes in and rescues Alex and Justin . Crumbs tells them to use their magic to stop Gorog . They arrive at Gorog 's lair and attempt to destroy him , but Mason tries to convince Alex to join them . Gorog also reveals Juliet ( young again ) , Justin 's ex-girlfriend , and she convinces Justin to join . Alex also surrenders , and Max is forced to as well , the three then use their combined magic to get rid of Gorog , and all the supernatural beings are freed from his influence . Juliet and Justin get back together . Alex tricks Harper into moving in the Russos ' basement by making her think all the wizards and supernatural creatures , with no home , are going to stay . Special guest stars : Gregg Sulkin as Mason , Bridgit Mendler as Juliet van Heusen Guest stars : Ian Abercrombie as Professor Crumbs , John Rubinstein as Gorog , Frank Pacheco as Felix , Fred Stoller as Dexter , Josh Sussman as Hugh Normous , Christopher Douglas Reed as Ogre Note : This is part 4 of a 4 - part arc . Absent : Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 104 25 `` Rock Around the Clock '' Guy Distad Robert Boesel November 4 , 2011 ( 2011 - 11 - 04 ) 414 3.8 The current landlord of the Russo 's building decides to sell the sub shop , evicting the Russos of the shop , as well as their apartment and lair , which contains their portal . Without the portal , the Russos will lose all contact with the Wizard world . So , the entire family ( along with Harper ) travel back to 1957 to stop Jerry 's father ( the previous owner of the shop ) from selling the shop to the landlord , so that they do not get evicted . Jerry 's father agrees not to sell the shop , and the Russos return to present day . However , they accidentally leave Harper behind when Harper is distracted by a woman with a poodle , and Harper suggests that the woman should put a picture of a poodle on her skirt , creating a poodle skirt , which were popular in the 1950s . Back in present day , the Russos ( who have not yet realized Harper 's absence ) find the sub shop obsolete , boarded up , and empty , with the subway car and the lair missing , too . The Russos realized that they have messed with the fabric of time , and something they did while time - traveling in 1957 must have created a ripple effect , and affected present day . While Alex suggests that they go back to 1957 to find out what they did wrong , Justin denies this , saying they should not solve the problem by doing exactly what caused the problem , however Theresa states that they will have to go back , when she realizes that they left Harper in 1957 . The Russos return to 1957 , where Jerry 's father reveals that he did not sell the restaurant to the landlord , however the restaurant failed to make enough business , and that he has been forced to close it . Justin realizes that when Jerry 's father lost the restaurant , he moved out of the building and since their family no longer lived in the building , the lair disappeared in present day . The Russos decide just to find Harper and go back to present day , and discover that Harper enrolled at Tribeca Prep as soon as they forgot her in 1957 , as Harper did not want to ruin her perfect attendance record . Justin , Alex , and Max go to Tribeca Prep to retrieve Harper , while Jerry and Theresa wait at the sub shop , however Harper decides to stay in the past , as she finds new popularity in 1957 that she does not experience in present day . While at Tribeca , Max introduces the high - five to several students . Alex realizes that if they make the sub shop the 1957 high school hangout , the sub shop will get better business and will not have to close . However , no one has any fun at the sub shop that night as the jukebox is broken . Just as everyone is about to leave , Max plugs his MP3 player into the jukebox and Harper uses her popularity to get the students dancing and having fun at the shop , and the business at the sub shop peaks . The Russos return to the present day , where the lair and the sub shop are back , and Jerry owns the entire building , while the landlord is now the owner of a Janitor service , instead of the building , and the high - five is now called a `` Max '' . Alex then remembers that they forgot Harper back in 1957 and they go back and get her . ( the Russos accidentally end up in 1977 twice in the episode , thanks to Justin ) Guest stars : Brendan Bradley as Hank , Carlo Michael Mancini as Lenny Hune 105 26 `` Harperella '' Victor Gonzalez Richard Goodman & Justin Varava November 18 , 2011 ( 2011 - 11 - 18 ) 427 3.5 The episode begins on Waverly Place where Harper is reading a storybook to a group of children when Alex pushes her away . Alex explains that it 's a Wizard Storybook , a magical storybook that has the power of turning what the person reads real . Then Harper is teleported into the loft where Theresa , her evil ( or `` wicked '' as she referred ) stepmother calls her Harperella ( a combination of Harper and Cinderella ) and forces her to do the chores . Then her evil stepbrothers , Justin and Max come down and give her laundry to her wash . Later Harper is washing the Waverly Substation when Alex appears , as she is Harper 's fairy godmother . Alex explains to Harper that the fairy tale has to run as it 's supposed to before things can go back to the way they were and then vanishes . Then Max comes in and says that the prince charming has decided to use the Substation to be the ballroom for the royal party . Harper says that she can go to the party but her stepfamily laughs at her . Back in the loft Harper summons Alex , she uses a spell to transform Harper 's bookdress into a princess gown and glass bowls into glass slippers . They go down ( through the door ) into the street and Alex turns the hot dog stand into a carriage to take Harper to the party . There she dances with prince charming , Zeke . At the stroke of midnight she goes outside leaving her glass slipper behind . Then , to both girls ' shock , Justin , Max and Zeke ( with pig noses ) rush inside the substation being chased by the big , bad wolf , Mason . Alex realizes that some pages are missing and Cinderella and three little pigs stories were blended together . The pig - turned boys and Harper rush into the loft being chased by Mason . While Alex finds the pages Harper has to keep Mason from eating Zeke ( who built the straw house ) , Alex eventually finds the book pages in a drawer along with other book pages . Alex tries a page from Max 's favourite magic trick book then David Copperfield comes into the room and does some magic tricks which impresses Harper . But another page that Alex tried was from Justin 's science book and a caveman appeared , but the caveman has a thing for Harper . So when Harper is impressed by the magic trick he is jealous and tries to attack David Copperfield . Alex finds the right pages and the story goes back to Cinderella . Zeke Charming comes in and Theresa tries to wear the slipper but fails . Then Harper tries it and fits . She and Zeke go back to the royal party where they dance and then the story ends and the world is returned to its normality . Last but not least , Harper was back in the real world and she lived happily ever after . Finally , Alex planned that Harper and Zeke dance because they were great when they danced in romance , so they practiced but they were really bad . The ball dance is turned into a clogging dance . Special guest stars : David Copperfield as himself , Gregg Sulkin as Mason Guest star : Dan Benson as Zeke Absent : David DeLuise as Jerry Russo 106 27 `` Who Will Be the Family Wizard '' Victor Gonzalez Vince Cheung & Ben Montanio ( Part 1 ) Todd J. Greenwald ( Part 2 ) January 6 , 2012 ( 2012 - 01 - 06 ) 428 -- 429 9.8 During a dinner that Alex made for the family , Professor Crumbs visits to reveal that with this selfless act , the Russo children can finally have their family wizard competition . Round one of the competition is a trivia round , in which questions related to magic are asked . During this round , Harper and Zeke appear . Zeke has accidentally smelled a purple substance in the lair that caused him to turn purple . This causes a griffin to kidnap him and Harper . Alex , Justin and Max use their three time - outs to go and rescue their friends . They make it back , only to discover that they have been gone too long and have been disqualified from the competition . They all return home , where the lair disappears and they all lose their magic , thus becoming mortals . Justin and Max begin to hate Alex because of her insistence on saving Zeke and Harper resulting in them losing . Convinced that the loss of their powers has ruined the family , Jerry decides to sell the sub shop . The kids go for a day without magic , during which , slowly gaining back trust in each other as they reopen and run the shop . Suddenly , they are returned to the area of the Wizard Competition , where the host reveals that the griffin attack ( which was actually meant for Professor Crumbs ) and the weeks they went without magic were tests to test their family bond . They enter the final round of the competition , a massive labyrinth . The first out of the maze will win . Justin makes it out first and is declared the Family Wizard . However , he decides he can not accept this because Alex was about to get out first : Justin had been trapped by magical vines and Alex , near the exit , came back to help . Alex is thus declared the true family wizard . Professor Crumbs , proud of Justin 's integrity in telling the truth , declares that he is going to retire as Headmaster of Wiz - Tech and declares Justin the new headmaster , granting him full wizard powers . Considering Max is the only Russo child that does not get powers or becomes a full wizard , Jerry thus decides to one day pass down the sub shop to Max , which he happily accepts . It ends with the Russo family hugging and gathering together with Alex stating that they are all finally happy at the same time . Special guest stars : Gregg Sulkin as Mason , Bridgit Mendler as Juliet van Heusen Guest stars : Dan Benson as Zeke , Ian Abercrombie as Professor Crumbs , Andy Kindler as Chancellor Tootietootie Note : This is an hour long special .
  Fact 0.0: The Wizards of Waverly Place season 4 episode list gives each episode's number, title, directors, writers, original air date, production code and U.S. viewers in millions.
  Fact 0.1: Episode 80, 'Alex Tells the World', was directed by Victor Gonzalez and written by Gigi McCreery and Perry Rein.
  Fact 0.2: Episode 80 aired November 12, 2010, with production code 401 and 4.1 million U.S. viewers.
  Fact 0.3: In 'Alex Tells the World', the Russos and Mason make it back home but their wizard powers are still gone.
  Fact 0.4: In 'Alex Tells the World', Alex reveals to reporters that they are wizards and that a group of wizards is held captive by the government.
  Fact 0.5: In 'Alex Tells the World', Professor Crumbs tells Alex that everything was a test she failed and orders her to Wizard Court for exposing wizardry.
  Fact 0.6: In 'Alex Tells the World', Justin is ordered to Wizard Court because he exposed magic to Agent Lamwood.
  Fact 0.7: Agent Lamwood was a creation from Professor Crumbs' mind.
  Fact 0.8: In 'Alex Tells the World', the judges rule Alex and Justin guilty.
  Fact 0.9: In 'Alex Tells the World', Professor Crumbs demotes Alex and Justin to level one in the Family Wizard Competition, behind Max by two levels.
  Fact 0.10: In 'Alex Tells the World', Max sets traps in the Sub Station for the government but falls for one of them.
  Fact 0.11: Gregg Sulkin plays Mason as a special guest star in 'Alex Tells the World'.
  Fact 0.12: Ian Abercrombie guest stars as Professor Crumbs in 'Alex Tells the World'.
  Fact 0.13: Episode 81, 'Alex Gives Up', was directed by Victor Gonzalez and written by Todd J. Greenwald, and aired November 27, 2010 with production code 402.
  Fact 0.14: In 'Alex Gives Up', Alex decides to give up the wizard competition to spend more time with Mason.
  Fact 0.15: In 'Alex Gives Up', Alex thinks giving up does not matter since Max is all the way in the lead.
  Fact 0.16: In 'Alex Gives Up', Chancellor Tootietootie tells Alex that non-wizards and werewolves cannot date.
  Fact 0.17: In 'Alex Gives Up', Max's picture on the front of Future Wizards magazine attracts the daughter of a family of Cucuys.
  Fact 0.18: Cucuys are mythical Latino monsters.
  Fact 0.19: In 'Alex Gives Up', the Cucuys invite Max to a dinner party on their yacht.
  Fact 0.20: In 'Alex Gives Up', Alex plans to invite Chancellor Tootietootie to the Cucuy party to get him to appeal the law against werewolves and mortals dating.
  Fact 0.21: In 'Alex Gives Up', Lisa Cucuy finds out Mason is a werewolf and decides she wants him instead of Max.
  Fact 0.22: In 'Alex Gives Up', Alex gets jealous when Lisa flirts with Mason.
  Fact 0.23: In 'Alex Gives Up', Alex says embarrassing things about Mason to everyone on the yacht so he will get mad enough to turn into a werewolf.
  Fact 0.24: In 'Alex Gives Up', Carlos and Julie Cucuy jump off the boat because Cucuys are scared of werewolves.
  Fact 0.25: In 'Alex Gives Up', Jerry jumps off the boat after Carlos and Julie.
  Fact 0.26: In 'Alex Gives Up', Alex tells Lisa she can get off the boat the easy way or her way.
  Fact 0.27: In 'Alex Gives Up', Lisa jumps off the boat.
  Fact 0.28: In 'Alex Gives Up', Chancellor Tootietootie says the case will not be appealed because Mason turning into a werewolf proved the point that they are dangerous.
  Fact 0.29: In 'Alex Gives Up', Alex and Mason have to break up to be friends.
  Fact 0.30: In 'Alex Gives Up', Harper finishes writing her first novel based on the werewolf-wizard relationship.
  Fact 0.31: In 'Alex Gives Up', Jerry accidentally grabs Harper's just-finished novel while trying to get out of the water, causing him to fall back in.
  Fact 0.32: In 'Alex Gives Up', Justin tutors a class of delinquent wizards to gain a level to re-enter the Wizard Competition.
  Fact 0.33: Gregg Sulkin plays Mason as a special guest star in 'Alex Gives Up'.
  Fact 0.34: Andy Kindler guest stars as Chancellor Tootietootie in 'Alex Gives Up'.
  Fact 0.35: Frank Pacheco guest stars as Felix in 'Alex Gives Up'.
  Fact 0.36: David Barrera guest stars as Carlos Cucuy in 'Alex Gives Up'.
  Fact 0.37: Roxana Brusso guest stars as Julie Cucuy in 'Alex Gives Up'.
  Fact 0.38: Samantha Boscarino guest stars as Lisa Cucuy in 'Alex Gives Up'.
  Fact 0.39: Kari Wahlgren guest stars as Helen in 'Alex Gives Up'.
  Fact 0.40: Episode 82, 'Lucky Charmed', was directed by Victor Gonzalez and written by Justin Varava, and aired December 10, 2010 with production code 403 and 3.0 million U.S. viewers.
  Fact 0.41: In 'Lucky Charmed', Harper tries teaching Alex how to live without magic.
  Fact 0.42: In 'Lucky Charmed', Alex gets in big trouble for crashing Mr. Laritate's car, causing a disagreement with Harper.
  Fact 0.43: As punishment in 'Lucky Charmed', Alex has to clean the school's garbage cans.
  Fact 0.44: In 'Lucky Charmed', Mr. Laritate forgives Alex after seeing how the crash influenced her.
  Fact 0.45: In 'Lucky Charmed', Alex realizes she will be okay without magic.
  Fact 0.46: In 'Lucky Charmed', Alex uses Mr. Laritate's lucky life-saving hula girl figurine, Leilani, for luck.
  Fact 0.47: In 'Lucky Charmed', Alex turns into the hula girl and guides Harper through her driving test.
  Fact 0.48: In 'Lucky Charmed', Harper passes her driving test.
  Fact 0.49: In 'Lucky Charmed', Jerry wants to pass the wizard family robe to Max, but Justin has a hard time letting go of it.
  Fact 0.50: Bill Chott guest stars as Mr. Laritate in 'Lucky Charmed'.
  Fact 0.51: Julio Oscar Mechoso guest stars as DMV Instructor in 'Lucky Charmed'.
  Fact 0.52: Maria Canals Barrera, who plays Theresa Russo, was absent from 'Lucky Charmed'.
  Fact 0.53: Episode 83, 'Journey to the Center of Mason', was directed by Robbie Countryman and written by Peter Dirksen, and aired December 17, 2010 with production code 404.
  Fact 0.54: In 'Journey to the Center of Mason', Alex can no longer date Mason after finding out she will lose her powers.
  Fact 0.55: In 'Journey to the Center of Mason', Alex and Mason decide to try being best friends instead.
  Fact 0.56: In 'Journey to the Center of Mason', Dean, Alex's ex-boyfriend, asks Alex to get back together with him.
  Fact 0.57: In 'Journey to the Center of Mason', Mason gets jealous of Dean.
  Fact 0.58: In 'Journey to the Center of Mason', Max tells Mason that Dean and Alex used to date.
  Fact 0.59: In 'Journey to the Center of Mason', Mason turns into a werewolf and eats Dean.
  Fact 0.60: In 'Journey to the Center of Mason', Alex, Justin and Max turn Justin's toy Captain Jim Bob submarine big and go in it.
  Fact 0.61: In 'Journey to the Center of Mason', Alex, Justin and Max shrink the submarine and go into Mason's mouth.
  Fact 0.62: Inside Mason, Alex, Max and Justin see Mason's thoughts, which are all of Alex.
  Fact 0.63: In 'Journey to the Center of Mason', Alex, Max and Justin get Dean out.
  Fact 0.64: In 'Journey to the Center of Mason', Alex tells Dean he was knocked out and the doctor's solution was for them all to dress up as cows.
  Fact 0.65: In 'Journey to the Center of Mason', Dean offers again to get back together with Alex, and she says no.
  Fact 0.66: In 'Journey to the Center of Mason', Alex gets back together with Mason.
  Fact 0.67: Gregg Sulkin plays Mason as a special guest star in 'Journey to the Center of Mason'.
  Fact 0.68: Daniel Samonas guest stars as Dean in 'Journey to the Center of Mason'.
  Fact 0.69: Frank Pacheco guest stars as Felix in 'Journey to the Center of Mason'.
  Fact 0.70: Shoshana Bush guest stars as Tyler in 'Journey to the Center of Mason'.
  Fact 0.71: Episode 84, 'Three Maxes and a Little Lady', was directed by Victor Gonzalez and written by Richard Goodman, and aired January 7, 2011 with production code 405 and 4.0 million U.S. viewers.
  Fact 0.72: In 'Three Maxes and a Little Lady', Alex joins Justin's wizard class to try to move up in the competition.
  Fact 0.73: In 'Three Maxes and a Little Lady', Max joins a sophisticated wizard society.
  Fact 0.74: In 'Three Maxes and a Little Lady', the society suggests Max move the wizard competition up to a few days away.
  Fact 0.75: In 'Three Maxes and a Little Lady', Max quits the society because he finds out they are taking advantage of him.
  Fact 0.76: In 'Three Maxes and a Little Lady', Alex and Justin transform into Max to change the date of the family wizard competition back to its original date, and succeed.
  Fact 0.77: At the end of 'Three Maxes and a Little Lady', Alex and Justin accidentally turn Max indefinitely into a little girl.
  Fact 0.78: Bailee Madison guest stars as Maxine in 'Three Maxes and a Little Lady'.
  Fact 0.79: Frank Pacheco guest stars as Felix in 'Three Maxes and a Little Lady'.
  Fact 0.80: Kari Wahlgren guest stars as Helen in 'Three Maxes and a Little Lady'.
  Fact 0.81: Shane Harper guest stars as Fidel in 'Three Maxes and a Little Lady'.
  Fact 0.82: Bailee Madison begins her recurring role as Maxine, the female Max, in 'Three Maxes and a Little Lady'.
  Fact 0.83: Jake T. Austin was absent in four other episodes and returns at the end of the tenth episode, 'Back to Max'.
  Fact 0.84: Episode 85, 'Daddy's Little Girl', was directed by Robbie Countryman and written by Gigi McCreery and Perry Rein, and aired January 21, 2011 with production code 406 and 3.4 million U.S. viewers.
  Fact 0.85: In 'Daddy's Little Girl', the parents pay more attention to Max, who is now Maxine, and Alex becomes jealous.
  Fact 0.86: In 'Daddy's Little Girl', Justin takes Maxine to her karate class, where Maxine kicks Justin's butt.
  Fact 0.87: In 'Daddy's Little Girl', Alex makes Jerry's favorite sandwich to win his love back.
  Fact 0.88: In 'Daddy's Little Girl', Maxine announces she beat Justin in karate, and Jerry falls for her cuteness again.
  Fact 0.89: In 'Daddy's Little Girl', Alex and Justin cast spells on Maxine to change her back, which only makes her cuter and furious.
  Fact 0.90: In 'Daddy's Little Girl', Maxine demands she kicks either Alex's or Justin's butt.
  Fact 0.91: In 'Daddy's Little Girl', Alex casts a spell on Justin to make him a karate master.
  Fact 0.92: In 'Daddy's Little Girl', Justin beats Maxine, and Maxine is hurt physically.
  Fact 0.93: In 'Daddy's Little Girl', Jerry and Theresa rush to Maxine's side.
  Fact 0.94: In 'Daddy's Little Girl', Justin leaves, and Alex leaves after him, heartbroken.
  Fact 0.95: In 'Daddy's Little Girl', Jerry tells Alex he fell for Maxine because she reminded him of Alex when she was little.
  Fact 0.96: In 'Daddy's Little Girl', Zeke thinks the Russos hate him, but it was really Harper's alibi to keep him away from the Sub Station.
  Fact 0.97: Bailee Madison guest stars as Maxine in 'Daddy's Little Girl'.
  Fact 0.98: Dan Benson guest stars as Zeke in 'Daddy's Little Girl'.
  Fact 0.99: Lanny Joon guest stars as Sensei in 'Daddy's Little Girl'.
  Fact 0.100: Jake T. Austin, who plays Max, was absent from 'Daddy's Little Girl'.
  Fact 0.101: Episode 86, 'Everything's Rosie for Justin', was directed by Victor Gonzalez and written by Justin Varava, and aired February 4, 2011 with production code 407 and 3.9 million U.S. viewers.
  Fact 0.102: In 'Everything's Rosie for Justin', Justin and his delinquent class have a wizard wand dance assessment to get them back in Wiztech.
  Fact 0.103: In 'Everything's Rosie for Justin', Rosie, a new student, is an angel.
  Fact 0.104: In 'Everything's Rosie for Justin', Justin falls madly in love with Rosie.
  Fact 0.105: In 'Everything's Rosie for Justin', Harper makes cards that customers get hole punched each time they buy a sandwich, and nine punches earn a free sandwich.
  Fact 0.106: In 'Everything's Rosie for Justin', one man punches his own card, demands a free sandwich, and brings dozens of people with pre-punched cards.
  Fact 0.107: In 'Everything's Rosie for Justin', Maxine uses her cuteness to convince customers not to exploit her parents.
  Fact 0.108: In 'Everything's Rosie for Justin', Justin kicks Rosie out of his class so it does not fail the assessment.
  Fact 0.109: In 'Everything's Rosie for Justin', Alex gets Rosie to rejoin the class and puts a copycat spell on her.
  Fact 0.110: In 'Everything's Rosie for Justin', a squirrel-frog attaches itself to Alex's foot during the assessment.
  Fact 0.111: In 'Everything's Rosie for Justin', Rosie reveals herself to be an angel during the assessment.
  Fact 0.112: In 'Everything's Rosie for Justin', the class fails the assessment.
  Fact 0.113: In 'Everything's Rosie for Justin', Justin starts dating Rosie.
  Fact 0.114: Bailee Madison guest stars as Maxine in 'Everything's Rosie for Justin'.
  Fact 0.115: Frank Pacheco guest stars as Felix in 'Everything's Rosie for Justin'.
  Fact 0.116: Leven Rambin guest stars as Rosie in 'Everything's Rosie for Justin'.
  Fact 0.117: Diane Delano guest stars as Penny in 'Everything's Rosie for Justin'.
  Fact 0.118: Chris Coppola guest stars as Customer in 'Everything's Rosie for Justin'.
  Fact 0.119: Cameron Sanders guest stars as Nelvis in 'Everything's Rosie for Justin'.
  Fact 0.120: Jake T. Austin, who plays Max, was absent from 'Everything's Rosie for Justin'.
  Fact 0.121: Episode 87, 'Dancing with Angels', was directed by Victor Gonzalez and written by Richard Goodman, and aired February 11, 2011 with production code 408 and 4.5 million U.S. viewers.
  Fact 0.122: In 'Dancing with Angels', Justin and Rosie start dating, but their date gets ruined, so they want a second date.
  Fact 0.123: In 'Dancing with Angels', Alex suggests an Angel Club in Los Angeles, but Rosie says only Angels are allowed.
  Fact 0.124: In 'Dancing with Angels', Maxine will not tell their parents as long as Alex brings her back an Ozzy Osbourne stone.
  Fact 0.125: In 'Dancing with Angels', Justin, Harper and Alex sneak into the Angel Club with Rosie.
  Fact 0.126: In 'Dancing with Angels', Zeedrik asks if they are angels, and they say they are.
  Fact 0.127: In 'Dancing with Angels', Theresa and Jerry force Maxine to join a pageant so she will tell them where the others are.
  Fact 0.128: In 'Dancing with Angels', Alex and Harper cannot fly, so Justin secretly uses his wand to give Harper flying powers.
  Fact 0.129: In 'Dancing with Angels', Zeedrik tests them by having them sing while playing the harp, and they fail.
  Fact 0.130: In 'Dancing with Angels', Alex, Justin and Harper are accepted as angels until the key-chain of Ozzy Osbourne's star reverts to normal.
  Fact 0.131: In 'Dancing with Angels', Zeedrik believes they are really angels of darkness, and everyone in the club runs off, even Rosie.
  Fact 0.132: In 'Dancing with Angels', Rosie comes back to comfort Justin, and her wings are shown to be black.
  Fact 0.133: In 'Dancing with Angels', Justin does not see Rosie's black wings because he is looking forward.
  Fact 0.134: In 'Dancing with Angels', Maxine wins the beauty pageant, and Jerry and Theresa celebrate.
  Fact 0.135: Bailee Madison guest stars as Maxine in 'Dancing with Angels'.
  Fact 0.136: Leven Rambin guest stars as Rosie in 'Dancing with Angels'.
  Fact 0.137: Ransford Doherty guest stars as Zeedrik in 'Dancing with Angels'.
  Fact 0.138: Jake T. Austin, who plays Max, was absent from 'Dancing with Angels'.
  Fact 0.139: Episode 88, 'Wizards vs. Angels', was a one-hour special directed by Victor Gonzalez and written by Vince Cheung and Ben Montanio, and aired February 18, 2011 with production codes 409 and 410 and 5.1 million U.S. viewers.
  Fact 0.140: In 'Wizards vs. Angels', Rosie is an Angel of Darkness and has been tricking Justin to make him evil.
  Fact 0.141: In 'Wizards vs. Angels', Alex finds out Rosie is an Angel of Darkness with the help of Tina, a guardian angel in training.
  Fact 0.142: In 'Wizards vs. Angels', Justin crosses over to the dark side with Rosie.
  Fact 0.143: In 'Wizards vs. Angels', Justin steals the moral compass and gives it to Gorog so Gorog can turn it from good to bad.
  Fact 0.144: In 'Wizards vs. Angels', Rosie tries to get Justin to run away because Gorog wants to destroy him.
  Fact 0.145: In 'Wizards vs. Angels', Justin refuses to leave and breaks his wand in halves.
  Fact 0.146: In 'Wizards vs. Angels', Alex tries to convince Justin that he is a good wizard.
  Fact 0.147: In 'Wizards vs. Angels', Maxine is enrolled in Tribeca Prep as part of a cousin exchange program.
  Fact 0.148: In 'Wizards vs. Angels', Harper insists Maxine have a slumber party, which becomes a disaster.
  Fact 0.149: In 'Wizards vs. Angels', Tina reveals that Rosie was a good angel, and Gorog corrupted her before her transport arrived.
  Fact 0.150: In 'Wizards vs. Angels', Alex and Tina go to the Dark Realm to save Justin.
  Fact 0.151: In 'Wizards vs. Angels', Alex borrows a pair of wings from the guardian angels.
  Fact 0.152: In 'Wizards vs. Angels', Alex takes the Moral Compass and tries to turn it to good, but Justin stops her.
  Fact 0.153: In 'Wizards vs. Angels', Rosie uses her good angel side, and they fly up.
  Fact 0.154: In 'Wizards vs. Angels', Justin turns the compass to good when they land.
  Fact 0.155: In 'Wizards vs. Angels', Tina and Alex take the compass and run.
  Fact 0.156: In 'Wizards vs. Angels', when Rosie and Justin kiss, Rosie turns into a good angel with white wings.
  Fact 0.157: In 'Wizards vs. Angels', Justin loses his dark wings.
  Fact 0.158: Bill Chott guest stars as Mr. Laritate in 'Wizards vs. Angels'.
  Fact 0.159: Bailee Madison guest stars as Maxine in 'Wizards vs. Angels'.
  Fact 0.160: Leven Rambin guest stars as Rosie in 'Wizards vs. Angels'.
  Fact 0.161: John Rubinstein guest stars as Gorog in 'Wizards vs. Angels'.
  Fact 0.162: China McClain guest stars as Tina in 'Wizards vs. Angels'.
  Fact 0.163: Amy Tolsky guest stars as Florence in 'Wizards vs. Angels'.
  Fact 0.164: Jake T. Austin, who plays Max, was absent from 'Wizards vs. Angels'.
  Fact 0.165: Episode 89, 'Back To Max', was directed by Guy Distad and written by Todd J. Greenwald, and aired March 11, 2011 with production code 411 and 3.7 million U.S. viewers.
  Fact 0.166: In 'Back To Max', Justin invites Professor Crumbs to check on his delinquent wizards class.
  Fact 0.167: In 'Back To Max', Maxine interrupts, and Alex and Justin try to find a way to turn Maxine back to Max.
  Fact 0.168: In 'Back To Max', Harper takes a role in the school's Spirit of America play, but is replaced by Maxine.
  Fact 0.169: Jake T. Austin returns as Max near the end of 'Back To Max'.
  Fact 0.170: Bailee Madison marks her final appearance as Maxine in 'Back To Max'.
  Fact 0.171: Bill Chott guest stars as Mr. Laritate in 'Back To Max'.
  Fact 0.172: Ian Abercrombie guest stars as Professor Crumbs in 'Back To Max'.
  Fact 0.173: Bailee Madison guest stars as Maxine in 'Back To Max'.
  Fact 0.174: Cameron Sanders guest stars as Nelvis in 'Back To Max'.
  Fact 0.175: LJ Benet guest stars as Little Crumbs in 'Back To Max'.
  Fact 0.176: Jackie Evancho guest stars as Choir Girl in 'Back To Max'.
  Fact 0.177: Episode 90, 'Zeke Finds Out', was directed by David DeLuise and written by Peter Dirksen, and aired April 8, 2011 with production code 412 and 3.0 million U.S. viewers.
  Fact 0.178: In 'Zeke Finds Out', Alex uses magic to make Zeke's tricks come true after she gets bored of his failing magic tricks.
  Fact 0.179: In 'Zeke Finds Out', Zeke tells Harper he believes he is a wizard and books a magic act at the sub station for a kid's birthday party.
  Fact 0.180: In 'Zeke Finds Out', Justin and Harper decide to leave everything up to Alex.
  Fact 0.181: In 'Zeke Finds Out', Harper tells Alex to tell Zeke the truth about wizards.
  Fact 0.182: In 'Zeke Finds Out', Alex is reluctant to expose wizardry because it could get her kicked out of the wizard competition forever.
  Fact 0.183: In 'Zeke Finds Out', Zeke breaks up with Harper after finding out she is keeping a secret from him.
  Fact 0.184: In 'Zeke Finds Out', Alex makes Harper disappear before she can expose wizardry to Zeke.
  Fact 0.185: In 'Zeke Finds Out', Justin freezes everyone except himself and Zeke.
  Fact 0.186: In 'Zeke Finds Out', Justin tells Zeke that he is a wizard and flashes him into their lair to unveil the secret.
  Fact 0.187: In 'Zeke Finds Out', Zeke promises he will never tell anyone about wizards.
  Fact 0.188: In 'Zeke Finds Out', Zeke and Harper reconcile, and Alex and Harper apologize and make up.
  Fact 0.189: Dan Benson guest stars as Zeke in 'Zeke Finds Out'.
  Fact 0.190: Maurice G. Smith guest stars as Big Mitch in 'Zeke Finds Out'.
  Fact 0.191: Maria Canals Barrera, who plays Theresa Russo, was absent from 'Zeke Finds Out'.
  Fact 0.192: David DeLuise, who plays Jerry Russo, was absent from 'Zeke Finds Out'.
  Fact 0.193: Episode 91, 'Magic Unmasked', was directed by David DeLuise and written by Justin Varava, and aired May 13, 2011 with production code 413 and 2.7 million U.S. viewers.
  Fact 0.194: In 'Magic Unmasked', a former legendary Luchador named Muy Macho visits the Sub Station.
  Fact 0.195: In 'Magic Unmasked', Alex finds out she was the reason Muy Macho quit lucha libre.
  Fact 0.196: In 'Magic Unmasked', Alex arranges a lucha libre match with Muy Macho and Jerry to restore his honor.
  Fact 0.197: In 'Magic Unmasked', Zeke asks too many questions about wizardry.
  Fact 0.198: In 'Magic Unmasked', Justin offers Zeke one magical wish if he promises to stop questioning him about magic.
  Fact 0.199: In 'Magic Unmasked', Zeke's wish is to defeat Muy Macho in the match.
  Fact 0.200: In 'Magic Unmasked', Max pretends to like Talia's hobbies but does not enjoy them.
  Fact 0.201: Dan Benson guest stars as Zeke in 'Magic Unmasked'.
  Fact 0.202: Valente Rodriguez guest stars as Muy Macho in 'Magic Unmasked'.
  Fact 0.203: McKaley Miller guest stars as Talia in 'Magic Unmasked'.
  Fact 0.204: Episode 92, 'Meet the Werewolves', was directed by Victor Gonzalez and written by David Henrie, and aired June 17, 2011 with production code 415 and 4.0 million U.S. viewers.
  Fact 0.205: In 'Meet the Werewolves', Alex asks to meet Mason's parents, and Mason shows her fake parents.
  Fact 0.206: In 'Meet the Werewolves', Mason takes Alex to the Autumn Moon Feast, a werewolf holiday, to meet his real parents.
  Fact 0.207: In 'Meet the Werewolves', Alex uses a spell to change herself and Harper into werewolves.
  Fact 0.208: In 'Meet the Werewolves', Alex tries to be the worst werewolf possible in front of Mason's parents.
  Fact 0.209: In 'Meet the Werewolves', Mason's parents like Alex as a werewolf.
  Fact 0.210: In 'Meet the Werewolves', Mason tells his parents that he truly loves Alex and that she is a wizard.
  Fact 0.211: In 'Meet the Werewolves', Alex changes back to a wizard and Mason turns back to human form.
  Fact 0.212: In 'Meet the Werewolves', Mason's parents try to eat Alex and Mason, and they make a near escape.
  Fact 0.213: In 'Meet the Werewolves', Max makes his parents' food taste like kid food.
  Fact 0.214: In 'Meet the Werewolves', Jerry and Theresa act like little kids after eating the food.
  Fact 0.215: In 'Meet the Werewolves', Justin uses a spell on a pizza to turn Jerry and Theresa back to their regular selves.
  Fact 0.216: In 'Meet the Werewolves', Max uses a spell on cookies to make them older, and they turn into teenagers.
  Fact 0.217: Gregg Sulkin plays Mason as a special guest star in 'Meet the Werewolves'.
  Fact 0.218: Harry Van Gorkum guest stars as Grant Greybeck in 'Meet the Werewolves'.
  Fact 0.219: Robin Riker guest stars as Linda Greybeck in 'Meet the Werewolves'.
  Fact 0.220: Marianne Muellerleile guest stars as Molly in 'Meet the Werewolves'.
  Fact 0.221: Eric Zuckerman guest stars as Bill in 'Meet the Werewolves'.
  Fact 0.222: Episode 93, 'Beast Tamer', was directed by Victor Gonzalez and written by Gigi McCreery and Perry Rein, and aired June 24, 2011 with production code 416 and 4.0 million U.S. viewers.
  Fact 0.223: In 'Beast Tamer', Alex, Justin and Max win four tickets to the Beast Bowl.
  Fact 0.224: In 'Beast Tamer', Beast Tamer Chase Riprock delivers the tickets, flirts with Alex, and invites the group on a pre-bowl tour.
  Fact 0.225: In 'Beast Tamer', Alex invites Mason to go with them, but he says he cannot.
  Fact 0.226: In 'Beast Tamer', Justin tries to tame the beast so Max can have his photo taken with it, and both are caught and banned from the event.
  Fact 0.227: In 'Beast Tamer', Chase tries to kiss Alex, but she rejects him and returns to the mortal world.
  Fact 0.228: In 'Beast Tamer', Alex finds Mason coming out of a hardware store, where he is evasive about his plans.
  Fact 0.229: In 'Beast Tamer', Mason is making an anniversary present for Alex, which is a sculpture of himself and Alex.
  Fact 0.230: In 'Beast Tamer', Harper tells Mason that Chase likes Alex and urges him to surprise her with the gift at the Beast Bowl.
  Fact 0.231: In 'Beast Tamer', Justin and Max return to the Beast Bowl disguised as beast taming clowns.
  Fact 0.232: In 'Beast Tamer', the beast escapes and snatches Alex, and Alex throws the whip to Chase.
  Fact 0.233: In 'Beast Tamer', Chase tames the beast, which drops Alex.
  Fact 0.234: In 'Beast Tamer', Mason reveals his gift to Alex.
  Fact 0.235: In 'Beast Tamer', Alex and Mason leave together with the sculpture.
  Fact 0.236: Gregg Sulkin plays Mason as a special guest star in 'Beast Tamer'.
  Fact 0.237: Nick Roux guest stars as Chase in 'Beast Tamer'.
  Fact 0.238: Maria Canals Barrera, who plays Theresa Russo, was absent from 'Beast Tamer'.
  Fact 0.239: David DeLuise, who plays Jerry Russo, was absent from 'Beast Tamer'.
  Fact 0.240: Episode 94, 'Wizard of the Year', was directed by Victor Gonzalez and written by Richard Goodman, and aired July 8, 2011 with production code 417 and 3.6 million U.S. viewers.
  Fact 0.241: In 'Wizard of the Year', Alex is crowned Wizard of the Year for saving the world from dark angels.
  Fact 0.242: In 'Wizard of the Year', Alex is reinstated in the Wizard Competition.
  Fact 0.243: In 'Wizard of the Year', a banquet honors Alex's achievement.
  Fact 0.244: In 'Wizard of the Year', Chase Riprock visits the lair to congratulate Alex.
  Fact 0.245: In 'Wizard of the Year', Keith Keith hounds them about being in a romantic relationship, and Alex says she already has a boyfriend.
  Fact 0.246: In 'Wizard of the Year', Keith Keith confirms that Chase and Alex are dating.
  Fact 0.247: In 'Wizard of the Year', Justin's video only contains him complaining about Alex.
  Fact 0.248: In 'Wizard of the Year', Mason stands Alex up on the night of the awards banquet.
  Fact 0.249: In 'Wizard of the Year', Chancellor TootieTootie tells Justin his display of affection and pride for Alex may reinstate him into the competition.
  Fact 0.250: In 'Wizard of the Year', Justin tries to destroy the tape.
  Fact 0.251: In 'Wizard of the Year', Chase sits at the Russos' table at the banquet.
  Fact 0.252: In 'Wizard of the Year', Mason sees Alex and Chase flirting and a short battle between Mason and Chase ensues.
  Fact 0.253: In 'Wizard of the Year', Alex breaks up the fight and chooses neither Mason nor Chase.
  Fact 0.254: In 'Wizard of the Year', Alex breaks up with Mason.
  Fact 0.255: Gregg Sulkin plays Mason as a special guest star in 'Wizard of the Year'.
  Fact 0.256: Andy Kindler guest stars as Chancellor Tootietootie in 'Wizard of the Year'.
  Fact 0.257: Nick Roux guest stars as Chase in 'Wizard of the Year'.
  Fact 0.258: Justin Guarini guest stars as Keith Keith in 'Wizard of the Year'.
  Fact 0.259: Episode 95, 'Misfortune at the Beach', was directed by Jody Margolin Hahn and written by Vince Cheung and Ben Montanio, and aired July 24, 2011 with production code 420 and 3.5 million U.S. viewers.
  Fact 0.260: In 'Misfortune at the Beach', the Russos go to the beach on an extremely hot day.
  Fact 0.261: In 'Misfortune at the Beach', the Russos come across Zelzar, a fortune-telling machine.
  Fact 0.262: In 'Misfortune at the Beach', Jerry warns the kids that Zelzar is from the wizard world and a wizard's fortune really comes true.
  Fact 0.263: In 'Misfortune at the Beach', Alex gets a fortune that says 'Say good-bye to your life.'
  Fact 0.264: In 'Misfortune at the Beach', Alex comes close to life-threatening situations multiple times.
  Fact 0.265: In 'Misfortune at the Beach', Alex, Justin, Max and Harper make a compromise with Zelzar: if they extract him from the machine and let him enjoy a day at the beach, he will take back Alex's fortune.
  Fact 0.266: In 'Misfortune at the Beach', Zelzar reveals he cannot take away Alex's fortune but can give it to the next person.
  Fact 0.267: In 'Misfortune at the Beach', a little girl gets Alex's fortune.
  Fact 0.268: In 'Misfortune at the Beach', a man from Random Prize Giveaway says 'Say good-bye to your life because we're randomly giving you one million dollars!'
  Fact 0.269: Michael Carbonaro guest stars as Zelzar in 'Misfortune at the Beach'.
  Fact 0.270: John J. York guest stars as Game Show Host in 'Misfortune at the Beach'.
  Fact 0.271: Episode 96, 'Wizards vs. Asteroid', was directed by Victor Gonzalez and written by Peter Dirksen, and aired August 19, 2011 with production code 418 and 3.6 million U.S. viewers.
  Fact 0.272: In 'Wizards vs. Asteroid', a huge asteroid is hurtling towards the earth.
  Fact 0.273: In 'Wizards vs. Asteroid', the Russos decide to hide in the wizard world, but Alex, Justin and Max go into space to destroy the asteroid.
  Fact 0.274: In 'Wizards vs. Asteroid', Alex finds out she is not graduating high school.
  Fact 0.275: In 'Wizards vs. Asteroid', a small part of the asteroid hits the school, giving Alex a chance to graduate.
  Fact 0.276: Dan Benson guest stars as Zeke in 'Wizards vs. Asteroid'.
  Fact 0.277: Bill Chott guest stars as Mr. Laritate in 'Wizards vs. Asteroid'.
  Fact 0.278: Kristina Hayes guest stars as Reporter in 'Wizards vs. Asteroid'.
  Fact 0.279: Episode 97, 'Justin's Back In', was directed by Guy Distad and written by Todd J. Greenwald, and aired August 26, 2011 with production code 419 and 3.2 million U.S. viewers.
  Fact 0.280: In 'Justin's Back In', Justin's class of delinquent wizards has completed their studies and can rejoin WizTech if they all pass their final evaluation.
  Fact 0.281: In 'Justin's Back In', all of the students fail the final evaluation, which Alex suspects is wrong.
  Fact 0.282: In 'Justin's Back In', the class had actually passed, but the historian who checked the exams was evil and lied about the scores.
  Fact 0.283: In 'Justin's Back In', the evil historian attacks Justin's class, and Justin stops him and saves Alex.
  Fact 0.284: In 'Justin's Back In', a student stops the historian by pulling the most powerful wand in the wizard world out of a crystal ball.
  Fact 0.285: In 'Justin's Back In', the student is a descendant of a famous wizard who was previously thought to have no living descendants.
  Fact 0.286: In 'Justin's Back In', Justin's class is readmitted into WizTech, and Justin is officially back in the family wizard competition.
  Fact 0.287: In 'Justin's Back In', Jerry and Theresa try to recreate Max's childhood accomplishments with the help of Harper.
  Fact 0.288: In 'Justin's Back In', during the credits it is revealed that Max had kept his favorite memories all along.
  Fact 0.289: Tim Conway guest stars as Cragmont, as special guest star, in 'Justin's Back In'.
  Fact 0.290: Ian Abercrombie guest stars as Professor Crumbs in 'Justin's Back In'.
  Fact 0.291: Frank Pacheco guest stars as Felix in 'Justin's Back In'.
  Fact 0.292: Cameron Sanders guest stars as Nelvis in 'Justin's Back In'.
  Fact 0.293: Episode 98, 'Alex the Puppetmaster', was directed by Victor Gonzalez and written by Gigi McCreery and Perry Rein, and aired September 16, 2011 with production code 421 and 3.0 million U.S. viewers.
  Fact 0.294: In 'Alex the Puppetmaster', Alex and Harper decide to do a puppet show to raise money for a new apartment, and Zeke and Justin help them.
  Fact 0.295: In 'Alex the Puppetmaster', Alex forgets to write the lines for the show.
  Fact 0.296: In 'Alex the Puppetmaster', Harper gets mad and does a show of her own, with puppets that act like the way Alex treats her.
  Fact 0.297: In 'Alex the Puppetmaster', Alex makes Justin and Zeke puppets for her show and steals Harper's crowd.
  Fact 0.298: In 'Alex the Puppetmaster', Alex apologizes to Harper using the puppets.
  Fact 0.299: In 'Alex the Puppetmaster', Talia's parents meet Theresa and Jerry for the first time, and it does not go well.
  Fact 0.300: In 'Alex the Puppetmaster', Justin puts a spell on Talia's parents to forget they ever met Theresa and Jerry.
  Fact 0.301: In 'Alex the Puppetmaster', Talia's parents meet Theresa and Jerry again and end up watching a 16-hour play together.
  Fact 0.302: Joely Fisher guest stars as Meg Robinson, as special guest star, in 'Alex the Puppetmaster'.
  Fact 0.303: Dan Benson guest stars as Zeke in 'Alex the Puppetmaster'.
  Fact 0.304: James Urbaniak guest stars as Rob Robinson in 'Alex the Puppetmaster'.
  Fact 0.305: McKaley Miller guest stars as Talia in 'Alex the Puppetmaster'.
  Fact 0.306: Episode 99, 'My Two Harpers', was directed by Victor Gonzalez and written by Manny Basanese, and aired September 30, 2011 with production code 422 and 3.0 million U.S. viewers.
  Fact 0.307: In 'My Two Harpers', Harper and Zeke spend so much time together that Alex decides to make a Harper clone.
  Fact 0.308: In 'My Two Harpers', Justin comes up with ideas to save money in the sub-shop.
  Fact 0.309: In 'My Two Harpers', Max starts a fast-food restaurant in the wizard world.
  Fact 0.310: Dan Benson guest stars as Zeke in 'My Two Harpers'.
  Fact 0.311: Maria Canals Barrera, who plays Theresa Russo, was absent from 'My Two Harpers'.
  Fact 0.312: Episode 100, 'Wizards of Apartment 13B', was directed by Guy Distad and written by Justin Varava, and aired October 7, 2011 with production code 423 and 3.5 million U.S. viewers.
  Fact 0.313: In 'Wizards of Apartment 13B', Alex and Harper receive a flyer for a hotel.
  Fact 0.314: In 'Wizards of Apartment 13B', the hotel's owner, Dexter, sells them Apartment 13B, which belongs to a floor for non-humans.
  Fact 0.315: In 'Wizards of Apartment 13B', Harper is forced to use a training wand that can only open things.
  Fact 0.316: In 'Wizards of Apartment 13B', Mason lives on the floor and will not stop trying to win Alex back.
  Fact 0.317: In 'Wizards of Apartment 13B', Alex and Harper throw a party that the whole floor attends, including Felix.
  Fact 0.318: In 'Wizards of Apartment 13B', Mason uses Harper's wand to open Alex's heart, but she flirts with an ugly giant.
  Fact 0.319: In 'Wizards of Apartment 13B', Mason makes Alex open her heart to Felix.
  Fact 0.320: In 'Wizards of Apartment 13B', Theresa and Jerry trick Justin into going to Alex's apartment.
  Fact 0.321: In 'Wizards of Apartment 13B', Justin fixes Mason's mistakes by resetting Harper's wand, and it is revealed that he misses Alex.
  Fact 0.322: In 'Wizards of Apartment 13B', Alex, freed from Mason's spell, is angry at Mason, and Mason leaves.
  Fact 0.323: In 'Wizards of Apartment 13B', Dexter turns into Gorog outside the door when no one is looking.
  Fact 0.324: In 'Wizards of Apartment 13B', Max appears in the apartment, and Alex and Harper kick him out.
  Fact 0.325: In 'Wizards of Apartment 13B', Gorog gives Max a flyer for a free camp, and Max accepts.
  Fact 0.326: 'Wizards of Apartment 13B' is part 1 of a 4-part arc.
  Fact 0.327: 'Wizards of Apartment 13B' is the 100th episode.
  Fact 0.328: Gregg Sulkin plays Mason as a special guest star in 'Wizards of Apartment 13B'.
  Fact 0.329: John Rubinstein guest stars as Gorog in 'Wizards of Apartment 13B'.
  Fact 0.330: Frank Pacheco guest stars as Felix in 'Wizards of Apartment 13B'.
  Fact 0.331: Fred Stoller guest stars as Dexter in 'Wizards of Apartment 13B'.
  Fact 0.332: Christopher Douglas Reed guest stars as Ogre in 'Wizards of Apartment 13B'.
  Fact 0.333: Episode 101, 'Ghost Roommate', was directed by David DeLuise and written by Richard Goodman, and aired October 14, 2011 with production code 424 and 4.1 million U.S. viewers.
  Fact 0.334: In 'Ghost Roommate', Alex and Harper have trouble paying the apartment bills, and Dexter recommends getting a roommate.
  Fact 0.335: In 'Ghost Roommate', Alex and Harper find a wealthy ghost named Lucy and let her move in.
  Fact 0.336: In 'Ghost Roommate', Justin orders a robot to help Zeke at the sub station.
  Fact 0.337: In 'Ghost Roommate', Dexter, who is Gorog in disguise, makes an evil robot to spy on Justin and pass information to Gorog.
  Fact 0.338: In 'Ghost Roommate', Lucy explains that her boyfriend Donny disappeared sixty years prior in a plane accident.
  Fact 0.339: In 'Ghost Roommate', Alex tries to set Lucy up with Justin, but he makes matters worse.
  Fact 0.340: In 'Ghost Roommate', Lucy goes on a long date with Mason, which makes Alex jealous.
  Fact 0.341: In 'Ghost Roommate', Alex helps Lucy locate Donny, who had been stranded on an island in the Bermuda Triangle.
  Fact 0.342: In 'Ghost Roommate', Alex realizes she cannot get back home because heavy magnetic currents interfere with her magic.
  Fact 0.343: In 'Ghost Roommate', Mason goes to save Alex after he finds out she left for the Bermuda Triangle.
  Fact 0.344: In 'Ghost Roommate', the robot attacks Justin and Zeke after they find it scanning the spell book.
  Fact 0.345: In 'Ghost Roommate', the robot gives Gorog the spell he needed that he got from the Russos.
  Fact 0.346: In 'Ghost Roommate', Mason and Alex get off the island using the Bermuda shorts given to them by Justin.
  Fact 0.347: Gregg Sulkin plays Mason as a special guest star in 'Ghost Roommate'.
  Fact 0.348: Dan Benson guest stars as Zeke in 'Ghost Roommate'.
  Fact 0.349: Fred Stoller guest stars as Dexter in 'Ghost Roommate'.
  Fact 0.350: Josh Sussman guest stars as Hugh Normous in 'Ghost Roommate'.
  Fact 0.351: Linsey Godfrey guest stars as Lucy in 'Ghost Roommate'.
  Fact 0.352: Travis Caldwell guest stars as Donny in 'Ghost Roommate'.
  Fact 0.353: Michael Carbonaro guest stars as Robot in 'Ghost Roommate'.
  Fact 0.354: 'Ghost Roommate' is part 2 of a 4-part arc.
  Fact 0.355: Jake T. Austin, Maria Canals Barrera and David DeLuise were absent from 'Ghost Roommate'.
  Fact 0.356: Episode 102, 'Get Along, Little Zombie', was directed by Victor Gonzalez and written by Peter Dirksen, and aired October 21, 2011 with production code 425 and 3.4 million U.S. viewers.
  Fact 0.357: In 'Get Along, Little Zombie', Alex and Harper run into Mr. Laritate, who also lives in the hotel.
  Fact 0.358: In 'Get Along, Little Zombie', Felix tells Justin he broke his wand, and Justin says it is fake because he saw batteries in it.
  Fact 0.359: In 'Get Along, Little Zombie', Justin concludes that someone stole Felix's wand.
  Fact 0.360: In 'Get Along, Little Zombie', Dexter shows a zombie to the hall for magical creatures.
  Fact 0.361: In 'Get Along, Little Zombie', Mason messes with the elevator buttons so he can spend more time with Alex, and they land on the second floor.
  Fact 0.362: In 'Get Along, Little Zombie', all the humans discover the mysterious thirteenth floor.
  Fact 0.363: In 'Get Along, Little Zombie', Alex lies that they accidentally got into a secret open house floor.
  Fact 0.364: In 'Get Along, Little Zombie', Mr. Laritate discovers the thirteenth floor and sees Alex, Harper and Mason.
  Fact 0.365: In 'Get Along, Little Zombie', the zombie bites Mr. Laritate and turns him into a zombie.
  Fact 0.366: In 'Get Along, Little Zombie', Alex looks for a spell to turn Mr. Laritate human again.
  Fact 0.367: In 'Get Along, Little Zombie', Mr. Laritate runs away to the substation.
  Fact 0.368: In 'Get Along, Little Zombie', Professor Crumbs questions Felix, who reveals the 13th floor, but Crumbs denies that a 13th floor exists.
  Fact 0.369: In 'Get Along, Little Zombie', Alex and Mason are forced into a dance while trying to sneak the zombie into the substation.
  Fact 0.370: In 'Get Along, Little Zombie', Harper causes a distraction, and they give Mr. Laritate a medicine from the wizard world that turns him human.
  Fact 0.371: In 'Get Along, Little Zombie', Mr. Laritate remembers nothing.
  Fact 0.372: In 'Get Along, Little Zombie', Alex kisses Mr. Laritate on the cheek, and Mason suggests they just be friends.
  Fact 0.373: In 'Get Along, Little Zombie', Alex gets back together with Mason.
  Fact 0.374: In 'Get Along, Little Zombie', Crumbs tells Felix the floor should not exist.
  Fact 0.375: In 'Get Along, Little Zombie', Justin tells them the floor does not exist.
  Fact 0.376: In 'Get Along, Little Zombie', Crumbs thinks there is evil going on.
  Fact 0.377: In 'Get Along, Little Zombie', Dexter comes with dark angel wings and is revealed to have stolen Felix's wand.
  Fact 0.378: In 'Get Along, Little Zombie', Dexter tricks Felix into turning to the dark side in return for the wand.
  Fact 0.379: In 'Get Along, Little Zombie', Felix casts a spell that traps them all on the floor.
  Fact 0.380: In 'Get Along, Little Zombie', Mr. Laritate is dancing back at Waverly Place while Jerry sells chili.
  Fact 0.381: In 'Get Along, Little Zombie', Mr. Laritate eats chili and gets water to get the burn out of his mouth.
  Fact 0.382: Gregg Sulkin plays Mason as a special guest star in 'Get Along, Little Zombie'.
  Fact 0.383: Bill Chott guest stars as Mr. Laritate in 'Get Along, Little Zombie'.
  Fact 0.384: Ian Abercrombie guest stars as Professor Crumbs in 'Get Along, Little Zombie'.
  Fact 0.385: Frank Pacheco guest stars as Felix in 'Get Along, Little Zombie'.
  Fact 0.386: Fred Stoller guest stars as Dexter in 'Get Along, Little Zombie'.
  Fact 0.387: Regan Burns guest stars as Everett in 'Get Along, Little Zombie'.
  Fact 0.388: Jane Carr guest stars as Martha St. Claire in 'Get Along, Little Zombie'.
  Fact 0.389: 'Get Along, Little Zombie' is part 3 of a 4-part arc.
  Fact 0.390: Jake T. Austin and Maria Canals Barrera were absent from 'Get Along, Little Zombie'.
  Fact 0.391: Episode 103, 'Wizards vs. Everything', was directed by Victor Gonzalez and written by Gigi McCreery and Perry Rein, and aired October 28, 2011 with production code 426 and 3.4 million U.S. viewers.
  Fact 0.392: In 'Wizards vs. Everything', the wizards of the 13th floor come out, and Alex explains Dexter's evil.
  Fact 0.393: In 'Wizards vs. Everything', Dexter reveals himself as Gorog and says he is trying to rebuild his army of darkness.
  Fact 0.394: In 'Wizards vs. Everything', Felix uses a spell that makes all the wizards from the 13th floor go with them, leaving Alex, Justin and Harper behind.
  Fact 0.395: In 'Wizards vs. Everything', Justin realizes only mortals can go through the trashcan.
  Fact 0.396: In 'Wizards vs. Everything', Harper arrives in the wizard lair after Alex convinces her to go through the trashcan.
  Fact 0.397: In 'Wizards vs. Everything', Gorog takes the kidnapped wizards to his evil lair to make them evil.
  Fact 0.398: In 'Wizards vs. Everything', Gorog tricks Hugh Normous into becoming one of the evil wizards in his army.
  Fact 0.399: In 'Wizards vs. Everything', Professor Crumbs fakes becoming evil when Gorog plans to make him evil.
  Fact 0.400: In 'Wizards vs. Everything', Harper contacts Max and tells him about Gorog.
  Fact 0.401: In 'Wizards vs. Everything', Max suggests they look for spells in the wizard books.
  Fact 0.402: In 'Wizards vs. Everything', Gorog makes the wizard army dig a hole to the atmosphere of the wizard world, so they can take over it.
  Fact 0.403: In 'Wizards vs. Everything', Gorog tricks Mason into joining his army after Mason refuses.
  Fact 0.404: In 'Wizards vs. Everything', Gorog makes Felix send a black hole to suck Alex and Justin in.
  Fact 0.405: In 'Wizards vs. Everything', Gorog gives the kidnapped wizards their wands, but Crumbs refuses to let them take over the wizard world.
  Fact 0.406: In 'Wizards vs. Everything', Felix gets rid of Crumbs on Gorog's order.
  Fact 0.407: In 'Wizards vs. Everything', Crumbs arrives at the wizard lair as the black hole nearly sucks Alex and Justin in.
  Fact 0.408: In 'Wizards vs. Everything', Max suggests they put a black hole in the wizard lair.
  Fact 0.409: In 'Wizards vs. Everything', Max goes in and rescues Alex and Justin.
  Fact 0.410: In 'Wizards vs. Everything', Crumbs tells them to use their magic to stop Gorog.
  Fact 0.411: In 'Wizards vs. Everything', Mason tries to convince Alex to join Gorog.
  Fact 0.412: In 'Wizards vs. Everything', Gorog reveals Juliet, young again, Justin's ex-girlfriend, and she convinces Justin to join.
  Fact 0.413: In 'Wizards vs. Everything', Alex surrenders, and Max is forced to as well.
  Fact 0.414: In 'Wizards vs. Everything', Alex, Max and Justin use their combined magic to get rid of Gorog.
  Fact 0.415: In 'Wizards vs. Everything', all the supernatural beings are freed from Gorog's influence.
  Fact 0.416: In 'Wizards vs. Everything', Juliet and Justin get back together.
  Fact 0.417: In 'Wizards vs. Everything', Alex tricks Harper into moving into the Russos' basement by making her think all the homeless wizards and supernatural creatures will stay.
  Fact 0.418: Gregg Sulkin plays Mason as a special guest star in 'Wizards vs. Everything'.
  Fact 0.419: Bridgit Mendler guest stars as Juliet van Heusen, as special guest star, in 'Wizards vs. Everything'.
  Fact 0.420: Ian Abercrombie guest stars as Professor Crumbs in 'Wizards vs. Everything'.
  Fact 0.421: John Rubinstein guest stars as Gorog in 'Wizards vs. Everything'.
  Fact 0.422: Frank Pacheco guest stars as Felix in 'Wizards vs. Everything'.
  Fact 0.423: Fred Stoller guest stars as Dexter in 'Wizards vs. Everything'.
  Fact 0.424: Josh Sussman guest stars as Hugh Normous in 'Wizards vs. Everything'.
  Fact 0.425: Christopher Douglas Reed guest stars as Ogre in 'Wizards vs. Everything'.
  Fact 0.426: 'Wizards vs. Everything' is part 4 of a 4-part arc.
  Fact 0.427: Maria Canals Barrera, who plays Theresa Russo, was absent from 'Wizards vs. Everything'.
  Fact 0.428: David DeLuise, who plays Jerry Russo, was absent from 'Wizards vs. Everything'.
  Fact 0.429: Episode 104, 'Rock Around the Clock', was directed by Guy Distad and written by Robert Boesel, and aired November 4, 2011 with production code 414 and 3.8 million U.S. viewers.
  Fact 0.430: In 'Rock Around the Clock', the landlord of the Russos' building decides to sell the sub shop, evicting the Russos from the shop, their apartment and their lair, which holds their portal.
  Fact 0.431: In 'Rock Around the Clock', without the portal the Russos lose all contact with the wizard world.
  Fact 0.432: In 'Rock Around the Clock', the Russo family and Harper travel back to 1957 to stop Jerry's father from selling the shop to the landlord.
  Fact 0.433: In 'Rock Around the Clock', Jerry's father agrees not to sell the shop, and the Russos return to present day.
  Fact 0.434: In 'Rock Around the Clock', the Russos accidentally leave Harper behind in 1957 when she is distracted by a woman with a poodle.
  Fact 0.435: In 'Rock Around the Clock', Harper suggests the woman put a picture of a poodle on her skirt, creating the poodle skirt.
  Fact 0.436: In 'Rock Around the Clock', back in present day the sub shop is obsolete, boarded up and empty, with the subway car and the lair missing.
  Fact 0.437: In 'Rock Around the Clock', the Russos realize they messed with the fabric of time.
  Fact 0.438: In 'Rock Around the Clock', Theresa realizes they left Harper in 1957 and says they will have to go back.
  Fact 0.439: In 1957, Jerry's father reveals he did not sell the restaurant, but the restaurant failed to make enough business and he was forced to close it.
  Fact 0.440: In 'Rock Around the Clock', Justin realizes that when Jerry's father lost the restaurant, he moved out of the building, so the lair disappeared in present day.
  Fact 0.441: In 'Rock Around the Clock', Harper enrolled at Tribeca Prep as soon as she was forgotten in 1957, because she did not want to ruin her perfect attendance record.
  Fact 0.442: In 'Rock Around the Clock', Harper decides to stay in the past because she finds new popularity in 1957.
  Fact 0.443: In 'Rock Around the Clock', Max introduces the high-five to several students at Tribeca.
  Fact 0.444: In 'Rock Around the Clock', Alex realizes that making the sub shop the 1957 high school hangout will improve business.
  Fact 0.445: In 'Rock Around the Clock', the jukebox is broken, and Max plugs his MP3 player into the jukebox.
  Fact 0.446: In 'Rock Around the Clock', Harper uses her popularity to get the students dancing and having fun at the shop, and business peaks.
  Fact 0.447: In 'Rock Around the Clock', the Russos return to present day, where the lair and the sub shop are back.
  Fact 0.448: In 'Rock Around the Clock', Jerry owns the entire building, and the landlord now owns a Janitor service instead.
  Fact 0.449: In 'Rock Around the Clock', the high-five is now called a Max.
  Fact 0.450: In 'Rock Around the Clock', Alex remembers they forgot Harper in 1957, and they go back and get her.
  Fact 0.451: In 'Rock Around the Clock', the Russos accidentally end up in 1977 twice, thanks to Justin.
  Fact 0.452: Brendan Bradley guest stars as Hank in 'Rock Around the Clock'.
  Fact 0.453: Carlo Michael Mancini guest stars as Lenny Hune in 'Rock Around the Clock'.
  Fact 0.454: Episode 105, 'Harperella', was directed by Victor Gonzalez and written by Richard Goodman and Justin Varava, and aired November 18, 2011 with production code 427 and 3.5 million U.S. viewers.
  Fact 0.455: In 'Harperella', Harper is reading a storybook to children on Waverly Place when Alex pushes her away.
  Fact 0.456: In 'Harperella', the Wizard Storybook has the power of turning what the person reads real.
  Fact 0.457: In 'Harperella', Harper is teleported into the loft, where Theresa, her wicked stepmother, calls her Harperella and makes her do chores.
  Fact 0.458: In 'Harperella', Justin and Max are Harper's evil stepbrothers.
  Fact 0.459: In 'Harperella', Alex appears as Harper's fairy godmother.
  Fact 0.460: In 'Harperella', Max says Prince Charming has decided to use the Substation as the ballroom for the royal party.
  Fact 0.461: In 'Harperella', Alex uses a spell to turn Harper's bookdress into a princess gown and glass bowls into glass slippers.
  Fact 0.462: In 'Harperella', Alex turns the hot dog stand into a carriage to take Harper to the party.
  Fact 0.463: In 'Harperella', Harper dances with Prince Charming, Zeke.
  Fact 0.464: In 'Harperella', at the stroke of midnight Harper leaves her glass slipper behind.
  Fact 0.465: In 'Harperella', Justin, Max and Zeke, with pig noses, rush into the substation being chased by the big bad wolf, Mason.
  Fact 0.466: In 'Harperella', Cinderella and the three little pigs stories were blended together because some pages were missing.
  Fact 0.467: In 'Harperella', Alex tries a page from Max's favorite magic trick book, and David Copperfield appears and does magic tricks.
  Fact 0.468: In 'Harperella', Alex tries a page from Justin's science book, and a caveman appears who has a thing for Harper.
  Fact 0.469: In 'Harperella', the caveman is jealous and tries to attack David Copperfield.
  Fact 0.470: In 'Harperella', Alex finds the right pages and the story goes back to Cinderella.
  Fact 0.471: In 'Harperella', Theresa tries to wear the slipper but fails, and Harper fits it.
  Fact 0.472: In 'Harperella', Harper and Zeke dance at the royal party, and the world returns to normal.
  Fact 0.473: In 'Harperella', Alex planned for Harper and Zeke to dance, but they were really bad, so the ball dance turned into a clogging dance.
  Fact 0.474: David Copperfield guest stars as himself in 'Harperella'.
  Fact 0.475: Gregg Sulkin plays Mason as a special guest star in 'Harperella'.
  Fact 0.476: Dan Benson guest stars as Zeke in 'Harperella'.
  Fact 0.477: David DeLuise, who plays Jerry Russo, was absent from 'Harperella'.
  Fact 0.478: Episode 106, 'Who Will Be the Family Wizard', was directed by Victor Gonzalez and written by Vince Cheung and Ben Montanio for Part 1 and Todd J. Greenwald for Part 2, and aired January 6, 2012 with production codes 428 and 429 and 9.8 million U.S. viewers.
  Fact 0.479: In 'Who Will Be the Family Wizard', Professor Crumbs visits during a dinner Alex made for the family and reveals the Russo children can finally have their family wizard competition.
  Fact 0.480: In 'Who Will Be the Family Wizard', round one of the competition is a trivia round about magic.
  Fact 0.481: In 'Who Will Be the Family Wizard', Zeke smelled a purple substance in the lair and turned purple.
  Fact 0.482: In 'Who Will Be the Family Wizard', a griffin kidnaps Zeke and Harper.
  Fact 0.483: In 'Who Will Be the Family Wizard', Alex, Justin and Max use their three time-outs to rescue their friends.
  Fact 0.484: In 'Who Will Be the Family Wizard', the Russo children are disqualified for being gone too long.
  Fact 0.485: In 'Who Will Be the Family Wizard', the Russos lose their magic and become mortals when the lair disappears.
  Fact 0.486: In 'Who Will Be the Family Wizard', Justin and Max begin to hate Alex.
  Fact 0.487: In 'Who Will Be the Family Wizard', Jerry decides to sell the sub shop.
  Fact 0.488: In 'Who Will Be the Family Wizard', the kids go a day without magic and reopen and run the shop.
  Fact 0.489: In 'Who Will Be the Family Wizard', the host reveals the griffin attack was meant for Professor Crumbs and the weeks without magic were tests of the family bond.
  Fact 0.490: In 'Who Will Be the Family Wizard', the final round of the competition is a massive labyrinth where the first out wins.
  Fact 0.491: In 'Who Will Be the Family Wizard', Justin makes it out first and is declared the Family Wizard.
  Fact 0.492: In 'Who Will Be the Family Wizard', Justin declines the title because Alex was about to get out first and came back to help him.
  Fact 0.493: In 'Who Will Be the Family Wizard', Alex is declared the true family wizard.
  Fact 0.494: In 'Who Will Be the Family Wizard', Professor Crumbs retires as Headmaster of WizTech and declares Justin the new headmaster, granting him full wizard powers.
  Fact 0.495: In 'Who Will Be the Family Wizard', Jerry decides to pass down the sub shop to Max one day, and Max happily accepts.
  Fact 0.496: In 'Who Will Be the Family Wizard', the Russo family ends the episode hugging and gathering together, with Alex saying they are all finally happy at the same time.
  Fact 0.497: Gregg Sulkin plays Mason as a special guest star in 'Who Will Be the Family Wizard'.
  Fact 0.498: Bridgit Mendler guest stars as Juliet van Heusen, as special guest star, in 'Who Will Be the Family Wizard'.
  Fact 0.499: Dan Benson guest stars as Zeke in 'Who Will Be the Family Wizard'.
  Fact 0.500: Ian Abercrombie guest stars as Professor Crumbs in 'Who Will Be the Family Wizard'.
  Fact 0.501: Andy Kindler guest stars as Chancellor Tootietootie in 'Who Will Be the Family Wizard'.
  Fact 0.502: 'Who Will Be the Family Wizard' is an hour-long special.
Sentence 1: In the episode , `` Three Maxes and a Little Lady '' , Alex and Justin accidentally turn Max into a little girl , Maxine , portrayed by Bailee Madison . She ends up making Alex jealous when their father prefers Maxine over Alex . Maxine remains until `` Back to Max '' . In the episode `` Everything 's Rosie for Justin '' , Justin falls in love with Rosie , a guardian angel , but at the end of `` Dancing with Angels '' she becomes an Angel of Darkness . In `` Zeke Finds Out '' , the Russos finally reveal to Zeke that they are wizards . In `` Wizard of the Year '' , Alex is crowned Wizard of the Year for saving the world from the angels of darkness , and is reinstated in the Wizards Competition . In `` Justin 's Back In '' , Justin is reinstated in the Wizard Competition , after his class of delinquent wizards passes .
  Fact 1.0: In the episode 'Three Maxes and a Little Lady', Alex and Justin accidentally turn Max into a little girl named Maxine.
  Fact 1.1: Maxine is portrayed by Bailee Madison.
  Fact 1.2: Maxine makes Alex jealous when their father prefers Maxine over Alex.
  Fact 1.3: Maxine remains a little girl until 'Back to Max'.
  Fact 1.4: In 'Everything's Rosie for Justin', Justin falls in love with Rosie, a guardian angel.
  Fact 1.5: At the end of 'Dancing with Angels', Rosie becomes an Angel of Darkness.
  Fact 1.6: In 'Zeke Finds Out', the Russos finally reveal to Zeke that they are wizards.
  Fact 1.7: In 'Wizard of the Year', Alex is crowned Wizard of the Year for saving the world from the angels of darkness.
  Fact 1.8: In 'Wizard of the Year', Alex is reinstated in the Wizards Competition.
  Fact 1.9: In 'Justin's Back In', Justin is reinstated in the Wizard Competition after his class of delinquent wizards passes.

Retrieved items:
<item 1>
Wizards of Waverly Place (season 4)
Guest stars : Bill Chott as Mr. Laritate , Ian Abercrombie as Professor Crumbs , Bailee Madison as Maxine , Cameron Sanders as Nelvis , LJ Benet as Little
Crumbs , Jackie Evancho as Choir Girl Note : Jake T. Austin returns as Max near the end of the episode . Bailee Madison marks her final appearance as Maxine in
this episode . 90 11 `` Zeke Finds Out '' David DeLuise Peter Dirksen April 8 , 2011 ( 2011 - 04 - 08 ) 412 3.0 When Alex gets bored of Zeke 's failing magic
tricks , she decides to use magic to make Zeke 's tricks come true . Everything goes downhill when Zeke tells Harper he believes he is a wizard and is now
booking a magic act at the sub station for a kid 's birthday party . Since Justin and Harper are tired of Alex ruining things all the time , they decide to
leave everything up to her .
</item 1>

<item 2>
Wizards of Waverly Place (season 4)
Mason goes to save Alex after he finds out that she had left for the Bermuda Triangle . The robot attacks Justin and Zeke after they find him scanning the spell
book . Mason and Alex manage to get off of the island using the Bermuda shorts given to them by Justin . The robot is shown giving Gorog the spell he needed
that he got from the Russos . Special guest star : Gregg Sulkin as Mason Guest stars : Dan Benson as Zeke , Fred Stoller as Dexter , Josh Sussman as Hugh
Normous , Linsey Godfrey as Lucy , Travis Caldwell as Donny , Michael Carbonaro as Robot Note : This is part 2 of a 4 - part arc . Absent : Jake T. Austin as
Max Russo , Maria Canals Barrera as Theresa Russo , David DeLuise as Jerry Russo 102 23 `` Get Along , Little Zombie '' Victor Gonzalez Peter Dirksen October 21
, 2011 ( 2011 - 10 - 21 ) 425 3.4 Alex and Harper are going to their apartment when they run into Mr. Laritate , who also lives in the hotel .
</item 2>

<item 3>
Wizards of Waverly Place (season 4)
In the end Mason 's parents like her as a werewolf and she says to him that they do not know the real her . She begins to leave and he says that he truly loves
Alex to his parents and that she is a wizard . She changes back to a wizard and Mason turns back to human form . After the parents find out they try to eat her
and Mason and Alex make a near escape . Meanwhile Max makes his parents ' food taste like kid food in order to make it taste good . Afterwards Jerry and Theresa
eat it , and as a result they act like little kids . Justin thinks he will have to fix it but Max comments on how he does not need help , as he will be the
future family wizard . Max uses another spell on cookies to make them older , but they turn into teenagers . Finally , Justin uses a spell on a pizza to turn
them back into their regular selves .
</item 3>

<item 4>
Wizards of Waverly Place (season 4)
Laritate , Frank Pacheco as Felix , Bridgit Mendler as Juliet van Heusen , Daniel Samonas as Dean Moriarty , Bailee Madison as Maxine Russo , Shane Harper as
Fidel , Dan Benson as Zeke Beakerman , John Rubinstein as Gorog , Andy Kindler as Chancellor Tootietootie , Fred Stoller as Dexter , Leven Rambin as Rosie ,
China Anne McClain as Tina , Kari Wahlgren as Helen , Cameron Sanders as Nelvis , Josh Sussman as Hugh Normous , Jackie Evancho as Herself , McKaley Miller as
Talia , Valente Rodriguez as Muy Macho , and Nick Roux as Chase Riprock . Cast ( edit ) Selena Gomez as Alex Russo David Henrie as Justin Russo Jake T. Austin
as Max Russo Jennifer Stone as Harper Finkle Maria Canals Barrera as Theresa Russo David DeLuise as Jerry Russo Episodes ( edit ) Main article : List of Wizards
of Waverly Place episodes This season consists of 27 episodes . Wizards of Waverly Place Season 4 episodes No . overall No.
</item 4>

<item 5>
Wizards of Waverly Place (season 4)
Justin takes Maxine to her ( Max 's ) karate class and Maxine kicks Justin 's butt . After the class , Alex decides to make Jerry 's favorite sandwich in order
to win his love back , but Maxine comes along with Justin , announcing that she beat Justin in karate , and Jerry immediately falls for her cuteness again .
More jealous than ever , Alex comes back with Justin to Maxine 's karate class with their parents , having a slight idea of knowing how to change Maxine back to
Max . They cast spells on Maxine , Justin using the spell he used on Max that turned him into Maxine , and Alex using the reverse spell . Unfortunately , it
only makes Maxine cuter , which makes her furious . Instead of going against the person she was supposed to , she demands that she kicks either Alex or Justin
's butt . Justin goes against her , but seeing that he would lose , Alex casts a spell on Justin to make him a karate master . Too fast for Maxine , Justin
beats her , and Maxine is hurt ( physically ) .
</item 5>

<item 6>
Wizards of Waverly Place (season 4)
Guest stars : Bill Chott as Mr. Laritate , Bailee Madison as Maxine , Leven Rambin as Rosie , John Rubinstein as Gorog , China McClain as Tina , Amy Tolsky as
Florence Note : This is a one - hour special . Absent : Jake T. Austin as Max Russo 89 10 `` Back To Max '' Guy Distad Todd J. Greenwald March 11 , 2011 ( 2011
- 03 - 11 ) 411 3.7 Justin invites Professor Crumbs to check on his delinquent wizards class , to see how their wizard training is progressing . During the
conversation , Maxine interrupts , causing Alex and Justin to panic since they do not know how to explain `` Maxine '' , and they try to find a way to turn
Maxine back to Max . Meanwhile , Harper takes a role in the school 's `` Spirit of America '' play , only to be replaced by Maxine . Guest stars : Bill Chott as
Mr.
</item 6>

<item 7>
Wizards of Waverly Place (season 4)
The class fails , but Justin starts dating Rosie . Guest stars : Bailee Madison as Maxine , Frank Pacheco as Felix , Leven Rambin as Rosie , Diane Delano as
Penny , Chris Coppola as Customer , Cameron Sanders as Nelvis Absent : Jake T. Austin as Max Russo 87 8 `` Dancing with Angels '' Victor Gonzalez Richard
Goodman February 11 , 2011 ( 2011 - 02 - 11 ) 408 4.5 Justin and Rosie start dating but when their date gets ruined , they want a second date . Alex suggests an
Angel Club in Los Angeles , but Rosie says that only Angels are allowed . Maxine will not tell their parents as long as Alex brings her back an Ozzy Osbourne
stone . After Justin , Harper and Alex sneak in along with Rosie , Zeedrik asks them if they are angels and they say they are . Back at the sub shop , Theresa
and Jerry question Maxine to find out where Justin , Alex and Harper are , and Maxine does not say , so they force her to join a Pageant in order for her to
tell them where they are .
</item 7>

<item 8>
Wizards of Waverly Place (season 4)
They suggest he move the wizard competition up and he moves it to a few days away . Max quits the society because he finds out that they are taking advantage of
him . Alex and Justin transform into Max in an attempt to change the date of the family wizard competition back to its original date and succeed . In the end of
the episode they accidentally turn Max indefinitely into a little girl . Guest stars : Bailee Madison as Maxine , Frank Pacheco as Felix , Kari Wahlgren as
Helen , Shane Harper as Fidel Note : Bailee Madison begins her recurring role as Maxine , the female Max in this episode . Jake T. Austin was absent in four
other episodes and returns at the end of the tenth episode `` Back to Max '' . 85 6 `` Daddy 's Little Girl '' Robbie Countryman Gigi McCreery & Perry Rein
January 21 , 2011 ( 2011 - 01 - 21 ) 406 3.4 With Max still a little girl , their parents pay more attention to Max ( Maxine ) , and Alex becomes jealous .
</item 8>

<item 9>
Wizards of Waverly Place (season 4)
However , it turns out to be a trap made by Gorog to capture the wizards who live in the same floor to make them join the dark side and rule the wizard world .
He is defeated and destroyed by Alex , Justin and Max who were using the Power of Three . During this , Alex and Mason get back together and , after a long time
, Justin finally gets back together with Juliet . Finally , in the hour - long series finale , Alex , Justin , and Max compete in the family wizard competition
. Alex wins and gains full wizardry ; Justin becomes a full wizard as well when Professor Crumbs reveals he is retiring as headmaster of WizTech and passes the
position to Justin . Jerry also decides to pass down the Waverly Sub Station to Max one day since he is the only child who is not a wizard anymore . The series
ends with hugging and Alex saying that they are all happy . Guest stars and recurring cast include : Gregg Sulkin as Mason Greybeck , Ian Abercrombie as
Professor Crumbs , David Barrera as Carlos Cucuy , Samantha Boscarino as Lisa Cucuy , Bill Chott as Mr.
</item 9>

<item 10>
Wizards of Waverly Place (season 4)
Too fast for Maxine , Justin beats her , and Maxine is hurt ( physically ) . Jerry and Theresa rush to her side , caring for Maxine more than they care that
Justin beat Maxine in karate . Justin leaves , and Alex leaves after him , heartbroken . Back at home , Jerry tells Alex that he fell for Maxine because she
reminded him of Alex when she was little and they bond again . Meanwhile , Zeke thinks the Russos hate him , but really it was just Harper 's alibi in order to
keep him away from the Sub Station so as not to make him suspicious about Max 's disappearance and Maxine 's appearance . Guest stars : Bailee Madison as Maxine
, Dan Benson as Zeke , Lanny Joon as Sensei Absent : Jake T. Austin as Max Russo 86 7 `` Everything 's Rosie for Justin '' Victor Gonzalez Justin Varava
February 4 , 2011 ( 2011 - 02 - 04 ) 407 3.9 Justin and his delinquent class have a wizard wand dance assessment to do which will get them back in Wiztech .
</item 10>
=== END ITEM id=dc45f4cd64c0b7b9 ===

=== ITEM id=25f32d80d78e343e ===
Gold sentences and facts:
Sentence 0: Taxpayers in the United States may face various penalties for failures related to Federal , state , and local tax matters . The Internal Revenue Service ( IRS ) is primarily responsible for charging these penalties at the Federal level . The IRS can assert only those penalties specified imposed under Federal tax law . State and local rules vary widely , are administered by state and local authorities , and are not discussed herein .
  Fact 0.0: Taxpayers in the United States may face various penalties for failures related to federal tax matters
  Fact 0.1: Taxpayers in the United States may face various penalties for failures related to state tax matters
  Fact 0.2: Taxpayers in the United States may face various penalties for failures related to local tax matters
  Fact 0.3: The Internal Revenue Service (IRS) is primarily responsible for charging these penalties at the federal level
  Fact 0.4: The IRS can assert only those penalties specified under federal tax law
  Fact 0.5: State and local tax penalty rules vary widely
  Fact 0.6: State and local tax penalty rules are administered by state and local authorities
  Fact 0.7: State and local tax penalty rules are not discussed in the source text

Retrieved items:
<item 1>
IRS penalties
U.S. corporations more than 25 % owned , directly or indirectly , by foreign persons must file Form 5472 to report such ownership and all transactions with
related parties . Failure to timely file carries a $10,000 penalty per required form . This penalty may be increased by $10,000 per month per form for continued
failure to file . In addition , taxpayers who fail to report changes in foreign taxes used as credits against Federal income tax may be subject to penalties .
U.S. citizen or resident taxpayers ( including entities ) who are beneficiaries of a foreign trust or make transfers of property to a foreign trust must report
information about the transfer and the trust or corporation . Failure to timely report on Form 3520 or Form 3520 - A may result in penalties of up to 35 % .
Similar transferors to foreign corporations failing to file Form 926 may face penalties of 10 % of the value of the transfer , up to $100,000 . Excise taxes as
penalties ( edit ) Federal excise taxes are imposed on a variety of goods and services . Some of these taxes require purchase of tax stamps or other evidence of
advance payment of tax .
</item 1>

<item 2>
IRS penalties
IRS penalties - wikipedia IRS penalties Jump to : navigation , search This article is part of a series on Taxation in the United States of America Federal
taxation ( show ) Alternative minimum tax Capital gains tax Corporate tax Estate tax Excise tax Gift tax Income tax Payroll tax Internal Revenue Service ( IRS )
Internal Revenue Code ( IRC ) IRS tax forms Revenue by state History Constitutional authority Taxpayer standing Court Protest Evasion State and local taxation (
show ) State income tax Property tax Sales tax Use tax Land value tax State tax levels Federal tax reform ( show ) Automated payment transaction tax 9 -- 9 -- 9
Competitive Tax Plan Efficient Taxation of Income FairTax Flat tax Hall -- Rabushka flat tax Kemp Commission Taxpayer Choice Act USA Tax Value added tax Border
- adjustment tax United States portal Taxpayers in the United States may face various penalties for failures related to Federal , state , and local tax matters
. The Internal Revenue Service ( IRS ) is primarily responsible for charging these penalties at the Federal level . The IRS can assert only those penalties
specified imposed under Federal tax law . State and local rules vary widely , are administered by state and local authorities , and are not discussed herein .
</item 2>

<item 3>
IRS penalties
Similar penalties may apply to an income tax return ( Form 1120S ) for an S corporation return . 100 % penalty on unpaid withholding taxes ( edit ) Employers
are required to withhold income and social security taxes from wages paid to employees , and to pay these amounts promptly to the government . A penalty of 100
% of the amount not paid over ( plus liability for paying the withheld amounts ) may be collected without judicial proceedings from each and every person who
had custody and control of the funds and did not make the payment to the government . This applies to company employees and officers as individuals , as well as
to companies themselves . There have been reported cases of the IRS seizing houses of those failing to pay over employee taxes . Penalties for failure to
provide foreign information ( edit ) Taxpayers who are shareholders of controlled foreign corporations must file Form 5471 with respect to each such controlled
foreign corporation . Penalties for failure to timely file are $10,000 to $50,000 per form , plus possible loss of foreign tax credits . U.S.
</item 3>

<item 4>
IRS penalties
Tax adviser penalties ( edit ) Penalties also apply to people who promote tax shelters or who fail to maintain and disclose lists of reportable transactions
their customers or clients for those transactions . These monetary penalties can be severe . Judicial appeal of penalties ( edit ) Most penalties are subject to
judicial review . However , the courts rarely modify assessment of the penalties and interest for underestimate or late payment . Further reading ( edit )
Meldman , Robert E. , Federal Taxation Practice and Procedure , ISBN 978 - 0 - 8080 - 1647 - 2 , Chapters 9 and 18 Shafiroff , Ira L. , Internal Revenue Service
Practice and Procedure Deskbook , Kindle edition ( no ISBN ) , or hardcover ISBN 978 - 0 - 87224 - 109 - 1 Pratt , Kulsrud , et al , Federal Taxation , 2006
Edition ISBN 978 - 0 - 7593 - 5175 - 2 , pages 2 -- 6 through 2 - 15 .
</item 4>

<item 5>
IRS penalties
Clair , 45 Amer . Fed . Tax Rep. 2d 80 - 1528 , 80 - 1 U.S. Tax Cas . ( CCH ) ¶ 9364 , ( N.D. Texas 1980 ) , in which the court upheld seizure of the taxpayer
's personal residence for failure to pay over withholding taxes payable by the company . Jump up ^ 26 USC 6038 . Jump up ^ 26 USC 6038A . Jump up ^ 26 USC 6689
. Jump up ^ 26 USC 6677 . Jump up ^ 26 USC 6038B . Jump up ^ Subtitle D of 26 USC . Jump up ^ Manufacturers Taxes , IRS Jump up ^ 26 USC 4971 - 4980G Jump up ^
26 USC 4911 , 4912 and 26 USC 4940 - 4948 . Jump up ^ 26 USC 7301 - 7328 . Jump up ^ 26 USC 7201 - 7217 . Jump up ^ 26 USC 6702 26 USC 6673 .
</item 5>

<item 6>
IRS penalties
Some of these taxes require purchase of tax stamps or other evidence of advance payment of tax . Some require collection of the tax by retailers . An assortment
of penalties apply to manufacturers and retailers not complying with the particular rules . In addition , certain penalties are imposed in the form of an excise
tax . Pension and benefit plans must pay a tax for a variety of failures . Charities and private foundations must pay an excise tax on prohibited transactions
and other failures . Tax fraud penalties ( edit ) Intentional filing of materially false tax returns is a criminal offence . A person convicted of committing
tax fraud , or aiding and abetting another in committing tax fraud , may be subject to forfeiture of property and / or jail time . Conviction and sentencing is
through the court system . Responsibility for prosecution falls to the U.S. Department of Justice , not the Internal Revenue Service . Penalties may be assessed
against tax protesters who raise arguments that income tax laws are not valid , or who otherwise file frivolous returns or court petitions .
</item 6>

<item 7>
IRS penalties
This penalty of 20 % or 40 % of the increase in tax is due in the case of substantial understatement of tax , substantial valuation misstatements , transfer
pricing adjustments , or negligence or disregard of rules or regulations . For example , a valuation overstatement can result in a 30 % penalty on the amount of
tax owed . Special rules apply for each of these types of errors under which the penalty may be waived . Penalties in connection with information returns ( edit
) Certain types of returns , called `` information returns , '' do not require payment of tax . These include forms filed by employers to report wages ( Form W
- 2 ) and businesses to report certain payments ( Form 1099 series instructions ) . The penalty for failures related to these forms is a dollar amount per form
not timely filed , and the amount of penalty increases with the degree of lateness . The current maximum penalty for these forms is $50 . Many of the forms must
be filed electronically , and filing on paper is considered non-filing . Late filing of returns of partnership income ( Form 1065 ) can result in penalties of
$195 per month per partner , up to a maximum of 12 months .
</item 7>

<item 8>
IRS penalties
Willis , Hoffman , et al , South - Western Federal Taxation , 2009 Edition ISBN 978 - 0 - 324 - 66052 - 4 , pages 26 -- 13 through 26 - 19 . References ( edit )
Jump up ^ Most of the penalty rules are in Chapters 68 through 75 of the Internal Revenue Code , sections 6651 through 7344 . 26 USC Subtitle F . Jump up ^ Alan
J. Tarr , J.D. , LL. M. ( Taxation ) & Pamela Jensen Drucker , J.D. , LL. M. ( Taxation ) , Civil Tax Penalties , U.S. Income Portfolios , Vol. 634 ( 2d ed.
2012 ) , Bloomberg BNA . Jump up ^ 26 USC 6654 . Jump up ^ 26 USC 6651 ( a ) ( 3 ) . Jump up ^ See IRS Tax Topic 653 . Jump up ^ 26 USC 6651 ( a ) ( 1 ) .
</item 8>

<item 9>
IRS penalties
Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines . Contents ( hide ) 1 Underestimate and late payment
penalties 2 Penalties for failure to timely file returns or timely pay tax 3 Accuracy related penalties 4 Penalties in connection with information returns 5 100
% penalty on unpaid withholding taxes 6 Penalties for failure to provide foreign information 7 Excise taxes as penalties 8 Tax fraud penalties 9 Tax adviser
penalties 10 Judicial appeal of penalties 11 Further reading 12 References Underestimate and late payment penalties ( edit ) Taxpayers are required to have
withholding of tax or make quarterly estimated tax payments before the end of the tax year . Because accurate estimation requires accurate prediction of the
future , taxpayers sometimes underestimate the amount due . The penalty for paying too little estimated tax or having too little tax withheld is computed with
interest on the amount that should have been , but was not , paid . Where a taxpayer has filed an income or excise tax return that shows a balance due but does
not pay that balance by the due date of the return ( without extensions ) , a different charge applies .
</item 9>

<item 10>
IRS penalties
Penalties may be monetary or may involve forfeiture of property . Criminal penalties may include jail time , but are imposed only by a federal judge after a
defendant is convicted . Most monetary penalties are based on the amount of tax not properly paid . Penalties may increase with the period of nonpayment . Some
penalties are fixed dollar amounts or fixed percentages of some measure required to be reported . Excise taxes used as penalties are imposed in the Code
sections relating to particular kinds of transactions . Some penalties may be waived or abated where the taxpayer shows reasonable cause for the failure .
Penalties apply for failures to file income tax returns or information returns , or for filing incorrect returns . Some penalties may be very minor . Penalties
apply for certain types of errors on tax returns , and may be substantial . Some penalties are imposed as excise taxes on particular transactions . Certain
other penalties apply for other types of failures . Certain acts may result in forfeiture of property of the taxpayer . There are over 150 kinds of civil
penalties in the U.S. Internal Revenue Code , ranging in severity which is reflected in the amount of the applicable fines .
</item 10>
=== END ITEM id=25f32d80d78e343e ===
