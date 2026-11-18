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

=== ITEM id=27bbea2e6b4be808 ===
Gold sentences and facts:
Sentence 0: Stone tools were made from a variety of stones . For example , flint and chert were shaped ( or chipped ) for use as cutting tools and weapons , while basalt and sandstone were used for ground stone tools , such as quern - stones . Wood , bone , shell , antler ( deer ) and other materials were widely used , as well . During the most recent part of the period , sediments ( such as clay ) were used to make pottery . Agriculture was developed and certain animals were domesticated as well .
  Fact 0.0: Stone tools were made from a variety of stones.
  Fact 0.1: Flint and chert were shaped, or chipped, for use as cutting tools and weapons.
  Fact 0.2: Basalt and sandstone were used for ground stone tools.
  Fact 0.3: Basalt and sandstone were used for ground stone tools such as quern-stones.
  Fact 0.4: Wood, bone, shell, and antler (deer) were widely used materials.
  Fact 0.5: Other materials were widely used as well.
  Fact 0.6: During the most recent part of the period, sediments such as clay were used to make pottery.
  Fact 0.7: Agriculture was developed.
  Fact 0.8: Certain animals were domesticated.

Retrieved items:
<item 1>
Stone Age
The 1981 film Quest for Fire by Jean - Jacques Annaud tells the story of a group of early homo sapiens searching for their lost fire . A 21st - century series ,
Chronicles of Ancient Darkness by Michelle Paver tells of two New Stone Age children fighting to fulfil a prophecy and save their clan . See also Homo Ice Age
Life timeline List of Stone Age art Megalith Nature timeline Pleistocene Prehistoric warfare Timeline of the Stone Age Notes ^ Jump up to :
https://web.archive.org/web/20100818123718/http://www.nhm.ac.uk/about-us/news/2010/august/oldest-tool-use-and-meat-eating-revealed75831.html Jump up ^ Ko ,
Kwang Hyun ( 2016 ) . `` Origins of human intelligence : The chain of tool - making and brain evolution '' ( PDF ) . Anthropological Notebooks. 22 ( 1 ) : 5 --
22 . Jump up ^ Barham & Mitchell 2008 , p. 106 Jump up ^ Barham & Mitchell 2008 , p.
</item 1>

<item 2>
Stone Age
The species who made the Pliocene tools remains unknown . Fragments of Australopithecus garhi , Australopithecus aethiopicus and Homo , possibly Homo habilis ,
have been found in sites near the age of the Gona tools . End of the Stone Age Innovation of the technique of smelting ore ended the Stone Age and began the
Bronze Age . The first most significant metal manufactured was bronze , an alloy of copper and tin , each of which was smelted separately . The transition from
the Stone Age to the Bronze Age was a period during which modern people could smelt copper , but did not yet manufacture bronze , a time known as the Copper Age
, or more technically the Chalcolithic , `` copper - stone '' age . The Chalcolithic by convention is the initial period of the Bronze Age . The Bronze Age was
followed by the Iron Age . The transition out of the Stone Age occurred between 6000 BCE and 2500 BCE for much of humanity living in North Africa and Eurasia .
</item 2>

<item 3>
Stone Age
Stone Age - wikipedia Stone Age Jump to : navigation , search For other uses , see Stone Age ( disambiguation ) . Ġgantija temples in Gozo , Malta . Some of the
world 's oldest free - standing structures . The Stone Age ↑ before Homo ( Pliocene ) Paleolithic Lower Paleolithic Late Stone Age Homo Control of fire Stone
tools Middle Paleolithic Middle Stone Age Homo neanderthalensis Homo sapiens Recent African origin of modern humans Upper Paleolithic Late Stone Age Behavioral
modernity , Atlatl , Origin of the domestic dog Epipaleolithic Mesolithic Microliths , Bow , Canoe Natufian Khiamian Tahunian Heavy Neolithic Shepherd Neolithic
Trihedral Neolithic Pre-Pottery Neolithic Neolithic Neolithic Revolution , Domestication Pottery Neolithic Pottery ↓ Chalcolithic The Stone Age was a broad
prehistoric period during which stone was widely used to make implements with an edge , a point , or a percussion surface . The period lasted roughly 3.4
million years and ended between 8700 BCE and 2000 BCE with the advent of metalworking .
</item 3>

<item 4>
Stone Age
The key anatomical and behavioral features required for tool manufacture , which are possessed only by Hominans , are the larger thumb and the ability to hold
by means of an assortment of grips . Food and drink Main articles : Paleolithic diet and Paleolithic diet and nutrition Food sources of the Palaeolithic hunter
- gatherers were wild plants and animals harvested from the environment . They liked animal organ meats , including the livers , kidneys and brains . Large
seeded legumes were part of the human diet long before the agricultural revolution , as is evident from archaeobotanical finds from the Mousterian layers of
Kebara Cave , in Israel . Moreover , recent evidence indicates that humans processed and consumed wild cereal grains as far back as 23,000 years ago in the
Upper Paleolithic . Near the end of the Wisconsin glaciation , 15,000 to 9,000 years ago , mass extinction of Megafauna such as the Wooly mammoth occurred in
Asia , Europe , North America and Australia . This was the first Holocene extinction event .
</item 4>

<item 5>
Stone Age
Starting in the grasslands of the rift , Homo erectus , the predecessor of modern humans , found an ecological niche as a tool - maker and developed a
dependence on it , becoming a `` tool equipped savanna dweller . '' Stone Age in archaeology Beginning of the Stone Age Obsidian projectile point The oldest
indirect evidence found of stone tool use is fossilised animal bones with tool marks ; these are 3.4 million years old and were found in the Lower Awash Valley
in Ethiopia . Archaeological discoveries in Kenya in 2015 , identifying possibly the oldest known evidence of hominin use of tools to date , have indicated that
Kenyanthropus platyops ( a 3.2 to 3.5 - million - year - old Pliocene hominin fossil discovered in Lake Turkana , Kenya in 1999 ) may have been the earliest
tool - users known . The oldest stone tools were excavated from the site of Lomekwi 3 in West Turkana , northwestern Kenya , and date to 3.3 million years old .
</item 5>

<item 6>
Stone Age
Its beginnings are roughly contemporaneous with the European Upper Paleolithic . It lasts until historical times and this includes cultures corresponding to
Mesolithic and Neolithic in other regions . Material culture Tools Stone tools were made from a variety of stones . For example , flint and chert were shaped (
or chipped ) for use as cutting tools and weapons , while basalt and sandstone were used for ground stone tools , such as quern - stones . Wood , bone , shell ,
antler ( deer ) and other materials were widely used , as well . During the most recent part of the period , sediments ( such as clay ) were used to make
pottery . Agriculture was developed and certain animals were domesticated as well . Some species of non-primates are able to use stone tools , such as the sea
otter , which breaks abalone shells with them . Primates can both use and manufacture stone tools . This combination of abilities is more marked in apes and men
, but only men , or more generally Hominans , depend on tool use for survival .
</item 6>

<item 7>
Stone Age
Stone Age artifacts include tools used by modern humans and by their predecessor species in the genus Homo , and possibly by the earlier partly contemporaneous
genera Australopithecus and Paranthropus . Bone tools were used during this period as well but are rarely preserved in the archaeological record . The Stone Age
is further subdivided by the types of stone tools in use . The Stone Age is the first of the three - age system of archaeology , which divides human
technological prehistory into three periods : The Stone Age The Bronze Age The Iron Age Contents ( hide ) 1 Historical significance 2 Stone Age in archaeology
2.1 Beginning of the Stone Age 2.2 End of the Stone Age 2.3 Concept of the Stone Age 2.4 Three - stage system 2.5 Problem of the transitions 3 Chronology 3.1
Three - age chronology 3.1. 1 Lower Paleolithic 3.1. 1.1 Oldowan in Africa 3.1. 1.2 Oldowan out of Africa 3.1. 1.3 Acheulean in Africa 3.1.
</item 7>

<item 8>
Stone Age
This was the first Holocene extinction event . It possibly forced modification in the dietary habits of the humans of that age and with the emergence of
agricultural practices , plant - based foods also became a regular part of the diet . A number of factors have been suggested for the extinction : certainly
over-hunting , but also deforestation and climate change . The net effect was to fragment the vast ranges required by the large animals and extinguish them
piecemeal in each fragment . Shelter and habitat Around 2 million years ago , Homo habilis is believed to have constructed the first man - made structure in
East Africa , consisting of simple arrangements of stones to hold branches of trees in position . A similar stone circular arrangement believed to be around
380,000 years old was discovered at Terra Amata , near Nice , France . ( Concerns about the dating have been raised , see Terra Amata ) . Several human habitats
dating back to the Stone Age have been discovered around the globe , including : A tent - like structure inside a cave near the Grotte du Lazaret , Nice ,
France .
</item 8>

<item 9>
Stone Age
Globally , societies were hunter - gatherers but evidence of regional identities begins to appear in the wide variety of stone tool types being developed to
suit very different environments . Epipaleolithic / Mesolithic Main articles : Epipaleolithic and Mesolithic The period starting from the end of the last ice
age , 10,000 years ago , to around 6,000 years ago was characterized by rising sea levels and a need to adapt to a changing environment and find new food
sources . The development of Mode 5 ( microlith ) tools began in response to these changes . They were derived from the previous Paleolithic tools , hence the
term Epipaleolithic , or were intermediate between the Paleolithic and the Neolithic , hence the term Mesolithic ( Middle Stone Age ) . The choice of a word
depends on exact circumstances and the inclination of the archaeologists excavating the site . Microliths were used in the manufacture of more efficient
composite tools , resulting in an intensification of hunting and fishing and with increasing social activity the development of more complex settlements , such
as Lepenski Vir . Domestication of the dog as a hunting companion probably dates to this period .
</item 9>

<item 10>
Stone Age
It covers the greatest portion of humanity 's time ( roughly 99 % of `` human technological history '' , where `` human '' and `` humanity '' are interpreted to
mean the genus Homo ) , extending from 2.5 or 2.6 million years ago , with the first documented use of stone tools by hominans such as Homo habilis , to the end
of the Pleistocene around 10,000 BCE . The Paleolithic era ended with the Mesolithic , or in areas with an early neolithisation , the Epipaleolithic . Lower
Paleolithic Main article : Lower Paleolithic At sites dating from the Lower Paleolithic Period ( about 2,500,000 to 200,000 years ago ) , simple pebble tools
have been found in association with the remains of what may have been the earliest human ancestors . A somewhat more sophisticated Lower Paleolithic tradition ,
known as the Chopper chopping - tool industry , is widely distributed in the Eastern Hemisphere . This tradition is thought to have been the work of the hominin
species named Homo erectus . Although no such fossil tools have yet been found , it is believed that H. erectus probably made tools of wood and bone as well as
stone .
</item 10>
=== END ITEM id=27bbea2e6b4be808 ===

=== ITEM id=d773594516bd76ef ===
Gold sentences and facts:
Sentence 0: Third Generation ( GK Series ) Overview Production 2013 -- present Model years 2014 -- present Assembly Yorii , Japan Sumaré , Brazil Ayutthaya , Thailand Pingtung , Taiwan Karawang , Indonesia Malacca , Malaysia Guangzhou , China ( Guangqi Honda ) Celaya , Guanajuato , Mexico ( from Feb 2014 ) Tapukara , India ( from Jan 2015 ) Designer Riku Wada ( 2010 ) Body and chassis Related Honda City Honda HR - V Honda Freed Honda Shuttle Acura CDX Powertrain Engine 1.2 L L12B i - VTEC I4 petrol 1.3 L L13B Earth Dreams i - VTEC I4 1.5 L L15A I4 ( petrol ) 1.5 L L15B1 I4 ( petrol ) 1.5 L L15B Earth Dreams i - VTEC I4 1.5 L i - DTEC diesel Transmission 5 - speed manual ( China Only ) 6 - speed manual 7 - speed semi-automatic ( DCT ) CVT automatic Dimensions Wheelbase 2,530 mm ( 99.6 in ) Length 3,955 mm ( 155.7 in ) Width 1,695 mm ( 66.7 in ) Height 1,525 -- 1,550 mm ( 60.0 -- 61.0 in ) Curb weight 1,020 -- 1,080 kg ( 2,249 -- 2,381 lb ) 1,080 -- 1,170 kg ( 2,381 -- 2,579 lb ) ( Hybrid )
  Fact 0.0: The Third Generation (GK Series) has production from 2013 to the present.
  Fact 0.1: The Third Generation (GK Series) has model years from 2014 to the present.
  Fact 0.2: The Third Generation (GK Series) is assembled in Yorii, Japan.
  Fact 0.3: The Third Generation (GK Series) is assembled in Sumaré, Brazil.
  Fact 0.4: The Third Generation (GK Series) is assembled in Ayutthaya, Thailand.
  Fact 0.5: The Third Generation (GK Series) is assembled in Pingtung, Taiwan.
  Fact 0.6: The Third Generation (GK Series) is assembled in Karawang, Indonesia.
  Fact 0.7: The Third Generation (GK Series) is assembled in Malacca, Malaysia.
  Fact 0.8: The Third Generation (GK Series) is assembled in Guangzhou, China (Guangqi Honda).
  Fact 0.9: The Third Generation (GK Series) is assembled in Celaya, Guanajuato, Mexico, from February 2014.
  Fact 0.10: The Third Generation (GK Series) is assembled in Tapukara, India, from January 2015.
  Fact 0.11: The Third Generation (GK Series) was designed by Riku Wada in 2010.
  Fact 0.12: The Third Generation (GK Series) is related to the Honda City.
  Fact 0.13: The Third Generation (GK Series) is related to the Honda HR-V.
  Fact 0.14: The Third Generation (GK Series) is related to the Honda Freed.
  Fact 0.15: The Third Generation (GK Series) is related to the Honda Shuttle.
  Fact 0.16: The Third Generation (GK Series) is related to the Acura CDX.
  Fact 0.17: The Third Generation (GK Series) has a 1.2 L L12B i-VTEC I4 petrol engine.
  Fact 0.18: The Third Generation (GK Series) has a 1.3 L L13B Earth Dreams i-VTEC I4 engine.
  Fact 0.19: The Third Generation (GK Series) has a 1.5 L L15A I4 petrol engine.
  Fact 0.20: The Third Generation (GK Series) has a 1.5 L L15B1 I4 petrol engine.
  Fact 0.21: The Third Generation (GK Series) has a 1.5 L L15B Earth Dreams i-VTEC I4 engine.
  Fact 0.22: The Third Generation (GK Series) has a 1.5 L i-DTEC diesel engine.
  Fact 0.23: The 5-speed manual transmission is available in China only.
  Fact 0.24: The Third Generation (GK Series) has a 6-speed manual transmission.
  Fact 0.25: The Third Generation (GK Series) has a 7-speed semi-automatic (DCT) transmission.
  Fact 0.26: The Third Generation (GK Series) has a CVT automatic transmission.
  Fact 0.27: The Third Generation (GK Series) has a wheelbase of 2,530 mm (99.6 in).
  Fact 0.28: The Third Generation (GK Series) has a length of 3,955 mm (155.7 in).
  Fact 0.29: The Third Generation (GK Series) has a width of 1,695 mm (66.7 in).
  Fact 0.30: The Third Generation (GK Series) has a height of 1,525 to 1,550 mm (60.0 to 61.0 in).
  Fact 0.31: The Third Generation (GK Series) has a curb weight of 1,020 to 1,080 kg (2,249 to 2,381 lb).
  Fact 0.32: The hybrid version of the Third Generation (GK Series) has a curb weight of 1,080 to 1,170 kg (2,381 to 2,579 lb).

Retrieved items:
<item 1>
Honda Fit
National Highway Traffic Safety Administration ( NHTSA ) crash test ratings . The engine is a 1.5 L ( 92 cu in ) i - VTEC 4 - cylinder with 117 horsepower at
6600 rpm mated to a five - speed manual transmission or a five - speed automatic transmission . The second generation Honda Fit went on sale in North America
around August 26 , 2008 , for the 2009 model year . It was considered the most expensive within the subcompact class when equipped as such , but was also
described as modestly priced . It was reported in September 2009 that Honda is considering producing the Fit in North America in response to the increasing
strength of the Japanese yen against the U.S. dollar . For the 2011 models , Vehicle Stability Assist , cruise control , remote entry and USB audio interface
became standard across all trim levels . In addition , the Fit Sport now comes with carpeted floor mats .
</item 1>

<item 2>
Honda Fit
525 -- 1,550 mm ( 60.0 -- 61.0 in ) Curb weight 1,020 -- 1,080 kg ( 2,249 -- 2,381 lb ) 1,080 -- 1,170 kg ( 2,381 -- 2,579 lb ) ( Hybrid ) The third generation
Fit retains the overall design concept of previous generations , notably the center - located fuel tank and multi-configuration interior concept , marketed as
Magic Seating . Honda 's all - new `` Global Small Car Platform '' employing ultra-high strength 780 MPa yield steel for 27 % of its bodywork -- and a shorter
overall length ( - 1.6 inches ) , longer wheelbase ( + 1.2 inch ) increased rear legroom ( + 4.8 inches ) and increased passenger volume ( + 4.9 cu ft )
compared to its previous generation . Body panels are both welded and bolted to the frame in a hybrid monocoque and spaceframe fusion -- and rear torsion beam
suspension is more compact , no longer using an anti-sway bar to maximize interior and cargo space . The HR - V shares its platform with the Fit .
</item 2>

<item 3>
Honda Fit
( April 28 , 2014 ) . `` Smaller , Larger '' . Autoweek. 64 ( 9 ) : 28 . Jump up ^ `` Honda Jazz 2014 specifications ( Thailand ) '' ( PDF ) . Jump up ^ `` 2015
US - spec Honda Fit launched in Detroit '' . worldcarfans . Retrieved 2014 - 04 - 16 . ^ Jump up to : `` New Honda Fit introduced with `` Earth Dreams '' engine
, CVT , new body construction `` . sae.org . Jump up ^ `` Honda Fit specs in Japan '' . Retrieved 2014 - 04 - 16 . Jump up ^ `` Honda Worldwide - September 5 ,
2013 `` Honda to Release all - New Fit and Fit Hybrid in Japan '' `` . honda.com . Archived from the original on December 2 , 2013 . Jump up ^
http://www.extremetech.com/extreme/174741-2015-honda-fit-brings-more-tech-to-the-subcompact-market-navigation-for-60-anyone The 2015 Fit was manufactured in
Honda 's new Mexico plant .
</item 3>

<item 4>
Honda Fit
Indonesia Santa Rosa , Philippines Guangzhou , China ( Guangqi Honda ) Sumaré , Brazil ( Honda Brazil ) Designer Mitsuhiro Honda ( 1999 ) Body and chassis
Related Honda Airwave Honda City / Fit Aria Powertrain Engine 1.2 L L12A I4 ( petrol ) 1.3 L L13A I4 ( petrol ) 1.5 L L15A I4 ( petrol ) Transmission 5 - speed
manual 5 - speed automatic CVT automatic Dimensions Wheelbase 2,450 mm ( 96.5 in ) Length 3,845 mm ( 151.4 in ) ( Worldwide ) 4,000 mm ( 157.4 in ) ( U.S. &
Canada ) Width 1,675 mm ( 65.9 in ) Height 1,525 mm ( 60.0 in ) Curb weight 1,084 kg ( 2,390 lb ) The Honda Fit debuted in June 2001 in Japan and immediately
became a big hit . At its introduction in 2001 , it won the Car of the Year Japan Award .
</item 4>

<item 5>
Honda Fit
3 L L13A I4 ( petrol ) 1.5 L L15A I4 ( petrol ) Transmission 5 - speed manual 6 - speed manual 5 - speed automatic CVT automatic ( Japan ) Dimensions Wheelbase
2,500 mm ( 98.4 in ) Length 3,985 mm ( 156.9 in ) ( 2009 -- 11 ) 4,100 mm ( 161.4 in ) ( 2012 -- 13 ) 4,410 mm ( 173.6 in ) ( Fit Shuttle ) Width 1,695 mm (
66.7 in ) Height 1,525 mm ( 60.0 in ) 1,540 mm ( 60.6 in ) ( Fit Shuttle ) Curb weight 1,070 kg ( 2,359 lb ) The second generation Fit debuted on 17 October
2007 at the Tokyo Motor Show . At its introduction in 2007 , it won the Car of the Year Japan Award for the second time . The vehicle offered a longer wheelbase
than its predecessor and is wider and longer overall . Overall height is unchanged , while interior height increased by . 3 in .
</item 5>

<item 6>
Honda Fit
Retrieved 2010 - 01 - 20 . Jump up ^ `` Honda Adds a 1.5 L Model to the Popular `` Fit '' Small Car Line - up `` . Honda . 2002 - 09 - 12 . Archived from the
original on 2016 - 03 - 02 . Retrieved 2010 - 01 - 20 . Jump up ^ `` Top Gear Survey 2006 - Honda Jazz '' . Archived from the original on 2007 - 10 - 11 .
Retrieved 2008 - 01 - 14 . Jump up ^ `` Honda Begins Production of `` Fit '' Hatchback in China `` . Honda . 2004 - 09 - 13 . Archived from the original on 2016
- 03 - 02 . Retrieved 2010 - 01 - 20 . Jump up ^ `` 2008 Honda Fit Review '' . JB car pages . Retrieved 2009 - 03 - 09 . Jump up ^ Chris Vander Doelen , Windsor
Star ( April 5 , 2007 ) .
</item 6>

<item 7>
Honda Fit
Gawker Media . Retrieved 2008 - 03 - 19 . Jump up ^ Bell , Damon . `` The Consumer Guide to 2009 Subcompact Cars '' . Consumer Guide Automotive . HowStuffWorks
. Archived from the original on 2009 - 08 - 30 . Retrieved 2009 - 08 - 25 . The most expensive subcompact , the top - trim - level Honda Fit with an optional
navigation system , has a still - modest base price of $18,760 . Jump up ^ `` Honda Fit production could shift to Ohio '' . The Columbus Dispatch . Archived
from the original on 2013 - 01 - 21 . Retrieved 2009 - 10 - 22 . Jump up ^ `` 2011 Honda Fit Receives More Standard Features ; VSA Added to All Models '' .
theautochannel.com . Jump up ^ `` 2009 Honda Fit Review '' . JB car pages . Retrieved 2009 - 03 - 09 . Jump up ^ `` 2012 Honda Fit Receives Updates and More
Features '' ( Press release ) .
</item 7>

<item 8>
Honda Fit
Honda . 2004 - 06 - 25 . Archived from the original on 2009 - 09 - 12 . Retrieved 2010 - 01 - 20 . Jump up ^ Nunez , Alex ( 2007 - 07 - 18 ) . `` Honda :
2,000,000 Fits sold worldwide '' . Autoblog . Retrieved 2007 - 07 - 18 . Jump up ^ Abuelsamid , Sam ( 2008 - 06 - 23 ) . `` Honda boosting Fit production to
meet U.S. demand '' . Autoblog . Retrieved 2010 - 03 - 20 . ^ Jump up to : `` American Honda December Sales '' . Hondanews.com . Archived from the original on
2010 - 02 - 12 . Retrieved 2010 - 10 - 07 . Jump up ^ `` American Honda December 2010 Sales '' . Hondanews.com. 2011 - 01 - 04 . Archived from the original on
2011 - 11 - 11 .
</item 8>

<item 9>
Honda Fit
A total of 56,432 Fits were sold in the United States in 2007 . The 2008 Honda Fit was largely unchanged from the 2007 with the exception of the TPMS and new
windshield wipers that abandon the large rubber blade design to a more traditional rigid plastic with thin rubber blade design . MSRP was also increased by
roughly US $100 . Total sales in the United States for the year were 79,794 , an increase of almost 41 % over previous year . Reception Honda marketed the Fit
as `` Small is the New Big '' . Car and Driver magazine described that `` ( n ) ot only is it endowed with an impressive array of standard features and an
upscale interior , but its dynamics rival the responses of cars with much fancier pedigrees and price tags . '' The magazine even cited its lane - change number
of 71.4 mph edged the best run by a Ferrari F430 Spider F1 tested . And the car was included in Car and Driver 's 10 Best Lists for three times . In May 2006 ,
Fit was placed first in a Car and Driver comparison of seven economy cars that included Nissan Versa and Toyota Yaris .
</item 9>

<item 10>
Honda Fit
Retrieved 2009 - 12 - 11 . Jump up ^ James Healey ( Feb 28 , 2011 ) . `` Consumer Reports lauds Honda Fit for value , rips Jeep Wrangler '' . USA Today . Jump
up ^ `` Honda Fit tops list of best value cars '' . The Boston Globe . Mar 25 , 2012 . Archived from the original on July 9 , 2012 . Jump up ^ `` 2012 Best Cars
for Families Awards '' . U.S. News Rankings and Reviews . Retrieved 2012 - 04 - 01 . External links ( edit ) Look up fit in Wiktionary , the free dictionary .
Wikimedia Commons has media related to Honda Fit , Honda Jazz and Honda Fit EV . Fit / Jazz at Honda Worldwide Honda Electric Vehicles Honda Fit Hybrid video at
Honda World site Honda recalls since 1977 Honda Motor Company Divisions and subsidiaries Acura American Honda Honda Canada Canada Manufacturing Aircraft
Performance Development India Motorcycles Racing F1 Taiwan China ( 65 % ) UK Joint ventures and shareholdings Bangladesh Honda Private Limited ( 70 % ) Dongfeng
Honda ( 50 % ) GE Honda Aero
</item 10>
=== END ITEM id=d773594516bd76ef ===

=== ITEM id=698549f659f9474b ===
Gold sentences and facts:
Sentence 0: Spike , occasionally referred to as Butch or Killer , is a stern but occasionally dumb American bulldog who is particularly disapproving of cats , but a softie when it comes to mice ( though in his debut appearance , Dog Trouble , Spike goes after both Tom and Jerry ) , and later , his son Tyke . In the shorts Jerry would often try to get Tom in trouble with Spike making him a shoo - in for a beating from the bulldog . Spike has a few weaknesses that Tom tries to capitalize upon : his possessiveness about his bone and his ticklishness . He made his first appearance in the 1942 Tom and Jerry cartoon Dog Trouble , and his first speaking role was in 1944 's The Bodyguard , where he was voiced by Billy Bletcher up until 1949 , from which point he was voiced by Daws Butler . Unlike his father Spike , Tyke does not speak . He only communicates by yapping , whimpering , growling , facial expressions and wagging his tail . In Tom and Jerry Kids , Tyke does have a speaking role in the program and is the first time that viewers were able to hear Tyke speak . Spike is very protective towards his son and gets very angry at Tom if Tyke is bothered or harmed . Although Tyke has spoken in Tom and Jerry Kids , he has laughed in one Tom and Jerry short . After Daws Butler , Maurice LaMarche , Frank Welker , John DiMaggio , Michael Donovan , Phil LaMarr and currently Rick Zieff would all perform Spike 's voice . Tyke 's vocal effects are provided by Frank Welker and speaking roles by Patric Zimmerman .
  Fact 0.0: Spike is occasionally referred to as Butch or Killer.
  Fact 0.1: Spike is a stern but occasionally dumb American bulldog.
  Fact 0.2: Spike is particularly disapproving of cats.
  Fact 0.3: Spike is a softie when it comes to mice.
  Fact 0.4: In his debut appearance, Dog Trouble, Spike goes after both Tom and Jerry.
  Fact 0.5: Spike later appears with his son, Tyke.
  Fact 0.6: In the shorts, Jerry would often try to get Tom in trouble with Spike.
  Fact 0.7: Jerry's trouble-making makes Tom a shoo-in for a beating from the bulldog Spike.
  Fact 0.8: Spike has a few weaknesses.
  Fact 0.9: Tom tries to capitalize on Spike's weaknesses.
  Fact 0.10: Spike's weaknesses include his possessiveness about his bone.
  Fact 0.11: Spike's weaknesses include his ticklishness.
  Fact 0.12: Spike made his first appearance in the 1942 Tom and Jerry cartoon Dog Trouble.
  Fact 0.13: Spike's first speaking role was in 1944's The Bodyguard.
  Fact 0.14: Billy Bletcher voiced Spike up until 1949.
  Fact 0.15: From 1949 onward, Spike was voiced by Daws Butler.
  Fact 0.16: Tyke does not speak, unlike his father Spike.
  Fact 0.17: Tyke only communicates by yapping.
  Fact 0.18: Tyke communicates by whimpering.
  Fact 0.19: Tyke communicates by growling.
  Fact 0.20: Tyke communicates by facial expressions.
  Fact 0.21: Tyke communicates by wagging his tail.
  Fact 0.22: Tyke has a speaking role in Tom and Jerry Kids.
  Fact 0.23: Tom and Jerry Kids was the first time viewers were able to hear Tyke speak.
  Fact 0.24: Spike is very protective towards his son Tyke.
  Fact 0.25: Spike gets very angry at Tom if Tyke is bothered or harmed.
  Fact 0.26: Tyke has spoken in Tom and Jerry Kids.
  Fact 0.27: Tyke has laughed in one Tom and Jerry short.
  Fact 0.28: After Daws Butler, Maurice LaMarche performed Spike's voice.
  Fact 0.29: After Daws Butler, Frank Welker performed Spike's voice.
  Fact 0.30: After Daws Butler, John DiMaggio performed Spike's voice.
  Fact 0.31: After Daws Butler, Michael Donovan performed Spike's voice.
  Fact 0.32: After Daws Butler, Phil LaMarr performed Spike's voice.
  Fact 0.33: Rick Zieff currently performs Spike's voice.
  Fact 0.34: Frank Welker provides Tyke's vocal effects.
  Fact 0.35: Patric Zimmerman provides Tyke's speaking roles.

Retrieved items:
<item 1>
List of Tom and Jerry characters
The most frequent love interest of Tom 's is Toodles Galore , who never has any dialogue in the cartoons . Despite five shorts ending with a depiction of Tom 's
apparent death , his demise is never permanent ; he even reads about his own death in a flashback in Jerry 's Diary . He appears to die in explosions in Mouse
Trouble ( after which he is seen in heaven ) , Yankee Doodle Mouse and in Safety Second , while in The Two Mouseketeers he is guillotined offscreen . Hanna -
Barbera era ( edit ) The following characters were introduced in the theatrical shorts that were directed by William Hanna and Joseph Barbera . Spike and Tyke (
edit ) Main article : Spike and Tyke ( characters ) Spike , occasionally referred to as Butch or Killer , is a stern but occasionally dumb American bulldog who
is particularly disapproving of cats , but a softie when it comes to mice ( though in his debut appearance , Dog Trouble , Spike goes after both Tom and Jerry )
, and later , his son Tyke .
</item 1>

<item 2>
Tom’s frequent love interest is Toodles Galore, who never speaks. Although several shorts depict Tom dying, his deaths are temporary. Hanna-Barbera introduced
Spike, a stern bulldog who usually dislikes cats but is gentle with mice and protective of his son, Tyke. Jerry often tricks Tom into provoking Spike, whose
weaknesses include his bone and ticklishness. Spike debuted in *Dog Trouble* (1942); Billy Bletcher and later Daws Butler voiced him. Tyke mostly
</item 2>

<item 3>
List of Tom and Jerry characters
`` Tom '' is a generic name for a male cat . He is usually but not always , portrayed as living a comfortable , or even pampered life , while Jerry ( named ``
Jinx '' in his debut appearance ) is a small , brown , house mouse who always lives in close proximity to Tom . Despite being very energetic , determined and
much larger , Tom is no match for Jerry 's wits . Jerry also possesses surprising strength for his size , approximately the equivalent of Tom 's , lifting items
such as anvils with relative ease and withstanding considerable impacts . Although cats typically chase mice to consume them , it is quite rare for Tom to
actually try to consume Jerry . Most of his attempts are just to torment or humiliate Jerry , sometimes in revenge , and sometimes to obtain a reward from a
human for catching Jerry . By the final `` fade - out '' of each cartoon , Jerry usually emerges triumphant , while Tom is shown as the loser . However , other
results may be reached . On rare occasions , Tom triumphs , usually when Jerry becomes the aggressor or when he pushes Tom a little too far .
</item 3>

<item 4>
The excerpt catalogs characters introduced across Tom and Jerry cartoons, television series, and films. It identifies Tom as a gray-and-white domestic shorthair
cat, originally named Jasper, and describes recurring or series-specific characters including Broke the vegetable-eating gopher; Robin Ho Ho’s merry men and the
Sheriff of Nottingham; Lightning Bolt the Super Squirrel, a flamboyant superhero; Morizzio, an amorous octopus; and a princess. Characters from the 2014 show
include Tom
</item 4>

<item 5>
List of Tom and Jerry characters
Trojan Dog ' in Tom and Jerry Comedy Show ) and movies thereafter . In his first appearance , the mad scientist is the owner of Jerry ( who is also his
assistant ) and has numerous cats trapped in a prison . When Jerry picks an orange cat named Lightning , the other cats flee in terror as the scientist pulls
him out for his latest experiment . He switches the mind of the orange cat with that of a bulldog and presents him as his present to Jerry . Thin lady ( edit )
An unnamed thin white lady became Tom 's owner in Buddies Thicker Than Water . Gene Deitch opted not to use Mammy 's character in his shorts , as he felt a ``
stereotypical black housekeep '' character `` did n't work in a modern context . '' Chuck Jones would also use her in The Unshrinkable Jerry Mouse and in the
re-animated version of Saturday Evening Puss . She also appeared in the movie Tom and Jerry : The Fast and the Furry . Like many other early human characters ,
her face was never shown .
</item 5>

<item 6>
The excerpt describes two characters from Chuck Jones-era Tom and Jerry shorts. The nameless, mouse-sized bulldog is Jerry’s pet and protector in *The Cat’s
Me-Ouch* and *Purr-Chance to Dream*; he resembles Spike, may be Spike’s eldest son, and has an unusually powerful bite. The Porpoise is a shark who attempts to
eat Tom and Jerry, appearing in *Cannery Rodent*, *Puss ’n’ Boats*, *File
</item 6>

<item 7>
The overview catalogs characters from *Tom and Jerry* shorts, television, films, games, and other media across multiple eras. Tom, a cat, repeatedly pursues the
clever, resourceful Jerry; their rivalry sometimes turns into cooperation or mutual rescue. Recurring characters include Spike and Tyke, Butch, Toodles,
Nibbles/Tuffy, Mammy Two Shoes, and numerous friends, relatives, rivals, pets, and one-off characters. The article also covers voice actors and character
</item 7>

<item 8>
List of Tom and Jerry characters
He is based on the character from Robin Hood tales . Yvonne jockalong ( edit ) Love interest to both Tom and Jerry , she appears in Sky Bunny and The Tennis
Menace . Sergeant farce ( edit ) A police officer who appears in See Dr. Jackal and Hide and in The Police Kitten where he is the superior officer to both Tom
and Jerry . Alley Cat Gang ( edit ) The Tom and Jerry Comedy Show ( edit ) The following characters were introduced in the television series The Tom and Jerry
Comedy Show . Droopy ( edit ) Main article : Droopy Dog A Basset Hound often with a miserable look on his face , originally introduced in the Tex Avery shorts
of the 1940s . He first made a poster cameo in Matinee Mouse . He also appeared in the Droopy and Dripple segments of Tom and Jerry Kids . He is sometimes an
adversary of Spike . He would also appear in Tom and Jerry Tales and would appear in the direct - to - video films .
</item 8>

<item 9>
List of Tom and Jerry characters
Like many other early human characters , her face was never shown . Chuck Jones era ( edit ) The following characters were introduced in the theatrical shorts
that were produced by Chuck Jones in the 1960s . Tiny bulldog ( edit ) This nameless mouse - sized bulldog , designed similarly to Spike the Bulldog , is Jerry
's pet dog companion in only two shorts : The Cat 's Me - Ouch and Purr - Chance to Dream . His role is similar to Spike in cartoons such as The Bodyguard , Fit
To Be Tied , and Much Ado About Mousing , which is to defend Jerry against Tom . He is presumably the oldest son of Spike before Tyke , and this could be
identified by his looks and personality that is similar to his father Spike . Unlike Spike and Tyke he has a big bite ( similar to those of Looney Tunes '
Tasmanian Devil ) . Porpoise ( edit ) A shark who tries to eat Tom and Jerry . He appears in Cannery Rodent , Puss ' n ' Boats , Filet Meow and Surf - Bored Cat
.
</item 9>

<item 10>
List of Tom and Jerry characters
In Tom and Jerry : Spy Quest , they are seen working for Doctor Zin . They also make an appearance in Tom and Jerry : Willy Wonka and the Chocolate Factory .
They were named after the Warner Bros. 1943 short Tin Pan Alley Cats . Tin is voiced by Greg Ellis , Pan by Jess Harnell and Alley by Richard McGonagle . Butch
( dog ) ( edit ) Main article : Butch Dog ( Tex Avery ) Butch ( Originally known as Spike , until renamed as Butch to avoid confusion with Spike from the Tom
and Jerry cartoons ) is the name of another bulldog from Tex Avery 's shorts in the 1940s and 1950s . He was one of the main antagonists in Droopy shorts (
although he was replaced by Hanna - Barbera 's Spike in the 1980s ) and also had a series of his own . He appears in a number of movies e.g. Tom and Jerry and
the Wizard of Oz and Tom and Jerry : The Magic Ring . References ( edit ) Jump up ^ Deitch , Gene ( 2015 ) .
</item 10>
=== END ITEM id=698549f659f9474b ===

=== ITEM id=b177f7a82f820f6c ===
Gold sentences and facts:
Sentence 0: Ladies ' singles at the XXIII Olympic Winter Games Venue Gangneung Ice Arena Gangneung , South Korea Dates 21 -- 23 February Competitors 30 from 20 nations Winning score 239.57 Medalists 01 ! Alina Zagitova Olympic Athletes from Russia 02 ! Evgenia Medvedeva Olympic Athletes from Russia 03 ! Kaetlyn Osmond Canada ← 2014 2022 →
  Fact 0.0: The Ladies' singles event was held at the XXIII Olympic Winter Games.
  Fact 0.1: The Ladies' singles event was held at the Gangneung Ice Arena in Gangneung, South Korea.
  Fact 0.2: The Ladies' singles event was held on 21 to 23 February.
  Fact 0.3: The Ladies' singles event had 30 competitors from 20 nations.
  Fact 0.4: The winning score in the Ladies' singles event was 239.57.
  Fact 0.5: Alina Zagitova of Olympic Athletes from Russia was the first-place medalist.
  Fact 0.6: Evgenia Medvedeva of Olympic Athletes from Russia was the second-place medalist.
  Fact 0.7: Kaetlyn Osmond of Canada was the third-place medalist.
  Fact 0.8: The navigation links point to the previous event in 2014 and the next event in 2022.
Sentence 1: The ladies ' single figure skating competition of the 2018 Winter Olympics was held at the Gangneung Ice Arena in Gangneung , South Korea . The short program was held on 21 February , and the free skating was held on 23 February .
  Fact 1.0: The ladies' single figure skating competition was held at the 2018 Winter Olympics.
  Fact 1.1: The ladies' single figure skating competition was held at the Gangneung Ice Arena in Gangneung, South Korea.
  Fact 1.2: The short program was held on 21 February.
  Fact 1.3: The free skating was held on 23 February.

Retrieved items:
<item 1>
Figure skating at the 2018 Winter Olympics – Ladies' singles
17 May 2016 . Retrieved 20 January 2018 . Jump up ^ `` Extract from ISU Communication 2097 '' . www.iceskating.org.uk/ . National Ice Skating Association ( UK )
. Retrieved 20 January 2018 . Jump up ^ Short program results Jump up ^ Free skating results Jump up ^ Final results Olympic champions in figure skating --
Ladies ' singles 1908 : Madge Syers ( GBR ) 1920 : Magda Julin ( SWE ) 1924 : Herma Szabo ( AUT ) 1928 : Sonja Henie ( NOR ) 1932 : Sonja Henie ( NOR ) 1936 :
Sonja Henie ( NOR ) 1948 : Barbara Ann Scott ( CAN ) 1952 : Jeannette Altwegg ( GBR ) 1956 : Tenley Albright ( USA ) 1960 : Carol Heiss ( USA ) 1964 : Sjoukje
Dijkstra ( NED ) 1968 : Peggy Fleming ( USA ) 1972 : Trixi Schuba ( AUT ) 1976 : Dorothy
</item 1>

<item 2>
Figure skating at the 2018 Winter Olympics – Ladies' singles
97 20 23 Ivett Tóth Hungary 150.43 53.22 23 97.21 23 24 Isadora Williams Brazil 144.18 55.74 17 88.44 24 Did not advance to free skating 25 Larkyn Austman
Canada N / A 51.42 25 N / A N / A 26 Diāna Ņikitina Latvia N / A 51.12 26 N / A N / A 27 Giada Russo Italy N / A 50.88 27 N / A N / A 28 Anita Östlund Sweden N
/ A 49.14 28 N / A N / A 29 Anna Khnychenkova Ukraine N / A 47.59 29 N / A N / A 30 Aiza Mambekova Kazakhstan N / A 44.40 30 N / A N / A References ( edit )
Jump up ^ `` Venues '' . www.pyeongchang2018.com/ . Pyeongchang 2018 Olympic Organizing Committee for the 2018 Winter Olympics .
</item 2>

<item 3>
Figure skating at the 2018 Winter Olympics – Ladies' singles
TES - Technical Element Score ; PCS - Program Component Score SS - Skating Skills ; TR - Transitions ; PE - Performance / Execution CH - Choreography ; IN -
Interpretation ; Ded - Deduction ; StN - Starting Number Free skating ( edit ) The free skating was held on 23 February . Pl . Name Nation TSS TES PCS SS TR PE
CH IN Ded StN Evgenia Medvedeva Olympic Athletes from Russia * 156.65 * 79.18 * 77.47 * 9.57 9.43 9.82 9.71 9.89 0.00 24 Alina Zagitova Olympic Athletes from
Russia * 156.65 * 81.62 * 75.03 * 9.32 9.21 9.61 9.29 9.46 0.00 22 Kaetlyn Osmond Canada 152.15 76.50 75.65 9.50 9.21 9.57 9.39 9.61 0.00 23 Satoko Miyahara
Japan 146.
</item 3>

<item 4>
Figure skating at the 2018 Winter Olympics – Ladies' singles
21 49.82 50.39 6.32 6.29 6.21 6.39 6.29 - 3.00 6 24 Isadora Williams Brazil 88.44 38.39 51.05 6.43 6.18 6.36 6.43 6.50 - 1.00 12 * Evgenia Medvedeva finished at
higher place due to better program component score Overall ( edit ) The skaters were ranked according to their overall score . TP - Total points ; SP - Short
program ; FS - Free skating Rank Name Nation Total points SP FS 01 ! Alina Zagitova Olympic Athletes from Russia 239.57 82.92 156.65 02 ! Evgenia Medvedeva
Olympic Athletes from Russia 238.26 81.61 156.65 03 ! Kaetlyn Osmond Canada 231.02 78.87 152.15 Satoko Miyahara Japan 222.38 75.94 146.44 5 Carolina Kostner
Italy 212.
</item 4>

<item 5>
Figure skating at the 2018 Winter Olympics – Ladies' singles
36 6.00 6.32 6.29 6.18 0.00 5 22 Maé - Bérénice Méité France 53.67 28.24 25.43 6.57 6.14 6.46 6.29 6.32 0.00 12 23 Ivett Tóth Hungary 53.22 27.60 25.62 6.54
6.18 6.43 6.39 6.50 0.00 14 24 Li Xiangning China 52.46 27.27 26.19 6.64 6.29 6.54 6.64 6.64 - 1.00 7 Did not advance to free skating 25 Larkyn Austman Canada
51.42 25.93 26.49 6.71 6.32 6.61 6.71 6.75 - 1.00 11 26 Diāna Ņikitina Latvia 51.12 26.35 24.
</item 5>

<item 6>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Contents ( hide ) 1 Records 2 Qualification 3 Schedule 4 Results 4.1 Short program 4.2 Free skating 4.3 Overall 5 References Records ( edit ) See also : World
and Olympic records set at the 2018 Winter Olympics For a complete list of figure skating records , see list of highest scores in figure skating . The following
new best scores were set during this competition : Component Skaters Score Date Ref Short program Evgenia Medvedeva ( OAR ) 81.61 21 February 2018 Alina
Zagitova ( OAR ) 82.92 The Olympic Athletes from Russia set two subsequent world records in the short program on February 21 , with Evgenia Medvedeva scoring
81.61 and then Alina Zagitova scoring 82.92 ten minutes later . Qualification ( edit ) Main article : Figure skating at the 2018 Winter Olympics --
Qualification A total of 30 skaters qualified to compete in the event , with each country allowed to only enter a maximum of three .
</item 6>

<item 7>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Figure skating at the 2018 Winter Olympics -- ladies ' singles - wikipedia Figure skating at the 2018 Winter Olympics -- ladies ' singles Jump to : navigation ,
search Ladies ' singles at the XXIII Olympic Winter Games Venue Gangneung Ice Arena Gangneung , South Korea Dates 21 -- 23 February Competitors 30 from 20
nations Winning score 239.57 Medalists 01 ! Alina Zagitova Olympic Athletes from Russia 02 ! Evgenia Medvedeva Olympic Athletes from Russia 03 ! Kaetlyn Osmond
Canada ← 2014 2022 → Figure skating at the 2018 Winter Olympics Qualification Singles men ladies Pairs mixed Ice dancing mixed Team event mixed The ladies '
single figure skating competition of the 2018 Winter Olympics was held at the Gangneung Ice Arena in Gangneung , South Korea . The short program was held on 21
February , and the free skating was held on 23 February .
</item 7>

<item 8>
Figure skating at the 2018 Winter Olympics – Ladies' singles
24 quotas were handed out during the 2017 World Figure Skating Championships and the remaining six were given out at the 2017 CS Nebelhorn Trophy . Each country
decided the entry of its teams , and athletes winning the quota were not necessarily granted the right to compete . All athletes competing must have met the
minimum total elements score , which does not include component scores . For the short program this was 20.00 and the free skate 36.00 . Schedule ( edit ) All
times are ( UTC + 9 ) . Date Time Round 21 February 10 : 00 Short program 23 February 10 : 00 Free skating Results ( edit ) Short program ( edit ) The short
program was held on 21 February . Pl . Name Nation TSS TES PCS SS TR PE CH IN Ded StN Alina Zagitova Olympic Athletes from Russia 82.92 WR 45.30 37.62 9.36 9.18
9.64 9.43 9.43 0.00 28 Evgenia Medvedeva Olympic Athletes from Russia 81.61 43.
</item 8>

<item 9>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Fleming ( USA ) 1972 : Trixi Schuba ( AUT ) 1976 : Dorothy Hamill ( USA ) 1980 : Anett Pötzsch ( GDR ) 1984 : Katarina Witt ( GDR ) 1988 : Katarina Witt ( GDR )
1992 : Kristi Yamaguchi ( USA ) 1994 : Oksana Baiul ( UKR ) 1998 : Tara Lipinski ( USA ) 2002 : Sarah Hughes ( USA ) 2006 : Shizuka Arakawa ( JPN ) 2010 : Yuna
Kim ( KOR ) 2014 : Adelina Sotnikova ( RUS ) 2018 : Alina Zagitova ( OAR ) Record scores in figure skating Combined total ( Total ) Short program ( SP ) Free
skating ( FS ) Short dance ( SD ) Free dance ( FD ) Juniors ( J ) Men Total : Yuzuru Hanyu 330.43 SP : Yuzuru Hanyu 112.72 FS : Yuzuru Hanyu 223.
</item 9>

<item 10>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Pyeongchang 2018 Olympic Organizing Committee for the 2018 Winter Olympics . Retrieved 26 December 2017 . ^ Jump up to : `` Schedule '' . Retrieved 6 January
2018 . Jump up ^ Start list ^ Jump up to : Bieler , Des ; Janes , Chelsea ( 21 February 2018 ) . `` Alina Zagitova tops Russian teammate Evgenia Medvedeva 's
world - record short program score '' . The Washington Post . ISSN 0190 - 8286 . Retrieved 21 February 2018 . ^ Jump up to : `` Russia 's Zagitova beats
Medvedeva 's short program world record at Olympics '' . TASS. 21 February 2018 . Retrieved 21 February 2018 . Jump up ^ `` Qualification Systems for XXIII
Olympic Winter Games , PyeongChang 2018 , Figure skating '' . International Skating Union ( ISU ) . 17 May 2016 . Retrieved 20 January 2018 .
</item 10>
=== END ITEM id=b177f7a82f820f6c ===

=== ITEM id=ecf8f5a7e1385f87 ===
Gold sentences and facts:
Sentence 0: The story of Shakespeare 's Hamlet was derived from the legend of Amleth , preserved by 13th - century chronicler Saxo Grammaticus in his Gesta Danorum , as subsequently retold by the 16th - century scholar François de Belleforest . Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur - Hamlet , though some scholars believe he himself wrote the Ur - Hamlet , later revising it to create the version of Hamlet we now have . He almost certainly wrote his version of the title role for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time . In the 400 years since its inception , the role has been performed by numerous highly acclaimed actors in each successive century .
  Fact 0.0: The story of Shakespeare's Hamlet was derived from the legend of Amleth.
  Fact 0.1: The legend of Amleth was preserved by the 13th-century chronicler Saxo Grammaticus in his Gesta Danorum.
  Fact 0.2: The legend of Amleth was subsequently retold by the 16th-century scholar François de Belleforest.
  Fact 0.3: Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur-Hamlet.
  Fact 0.4: Some scholars believe Shakespeare himself wrote the Ur-Hamlet and later revised it to create the version of Hamlet we have today.
  Fact 0.5: Shakespeare almost certainly wrote his version of the title role for his fellow actor Richard Burbage.
  Fact 0.6: Richard Burbage was the leading tragedian of Shakespeare's time.
  Fact 0.7: In the 400 years since the role of Hamlet was created, it has been performed by numerous highly acclaimed actors.
  Fact 0.8: The role of Hamlet has been performed by highly acclaimed actors in each successive century for 400 years.

Retrieved items:
<item 1>
Hamlet
Jenkins suggests that any personal satire may be found in the name `` Polonius '' , which might point to a Polish or Polonian connection . G.R. Hibbard
hypothesised that differences in names ( Corambis / Polonius : Montano / Raynoldo ) between the First Quarto and other editions might reflect a desire not to
offend scholars at Oxford University . Date ( edit ) John Barrymore as Hamlet ( 1922 ) `` Any dating of Hamlet must be tentative '' , cautions the New Cambridge
editor , Phillip Edwards . The earliest date estimate relies on Hamlet 's frequent allusions to Shakespeare 's Julius Caesar , itself dated to mid-1599 . The
latest date estimate is based on an entry , of 26 July 1602 , in the Register of the Stationers ' Company , indicating that Hamlet was `` latelie Acted by the
Lo : Chamberleyne his servantes '' . In 1598 , Francis Meres published his Palladis Tamia , a survey of English literature from Chaucer to its present day ,
within which twelve of Shakespeare 's plays are named .
</item 1>

<item 2>
Hamlet
Jr. , in the American Company 's production in Philadelphia in 1759 . David Garrick 's iconic hand gesture expresses Hamlet 's shock at the first sight of the
ghost ( artist : unknown ) . John Philip Kemble made his Drury Lane debut as Hamlet in 1783 . His performance was said to be 20 minutes longer than anyone else
's , and his lengthy pauses provoked the suggestion by Richard Brinsley Sheridan that `` music should be played between the words '' . Sarah Siddons was the
first actress known to play Hamlet ; many women have since played him as a breeches role , to great acclaim . In 1748 , Alexander Sumarokov wrote a Russian
adaptation that focused on Prince Hamlet as the embodiment of an opposition to Claudius 's tyranny -- a treatment that would recur in Eastern European versions
into the 20th century . In the years following America 's independence , Thomas Apthorpe Cooper , the young nation 's leading tragedian , performed Hamlet among
other plays at the Chestnut Street Theatre in Philadelphia , and at the Park Theatre in New York .
</item 2>

<item 3>
Hamlet
It was probably one of Shakespeare 's most popular works during his lifetime , and still ranks among his most performed , topping the performance list of the
Royal Shakespeare Company and its predecessors in Stratford - upon - Avon since 1879 . It has inspired many other writers -- from Johann Wolfgang von Goethe and
Charles Dickens to James Joyce and Iris Murdoch -- and has been described as `` the world 's most filmed story after Cinderella '' . The story of Shakespeare 's
Hamlet was derived from the legend of Amleth , preserved by 13th - century chronicler Saxo Grammaticus in his Gesta Danorum , as subsequently retold by the 16th
- century scholar François de Belleforest . Shakespeare may also have drawn on an earlier Elizabethan play known today as the Ur - Hamlet , though some scholars
believe he himself wrote the Ur - Hamlet , later revising it to create the version of Hamlet we now have . He almost certainly wrote his version of the title
role for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time .
</item 3>

<item 4>
Hamlet
It was immortalised both on record and on a film that played in US theatres for a week in 1964 as well as being the subject of books written by cast members
William Redfield and Richard L. Sterne . Other New York portrayals of Hamlet of note include that of Ralph Fiennes 's in 1995 ( for which he won the Tony Award
for Best Actor ) -- which ran , from first preview to closing night , a total of one hundred performances . About the Fiennes Hamlet Vincent Canby wrote in The
New York Times that it was `` ... not one for literary sleuths and Shakespeare scholars . It respects the play , but it does n't provide any new material for
arcane debates on what it all means . Instead it 's an intelligent , beautifully read ... '' Stacy Keach played the role with an all - star cast at Joseph Papp
's Delacorte Theatre in the early 70s , with Colleen Dewhurst 's Gertrude , James Earl Jones 's King , Barnard Hughes 's Polonius , Sam Waterston 's Laertes and
Raúl Juliá 's Osric .
</item 4>

<item 5>
Hamlet
London , 1776 The play was revived early in the Restoration . When the existing stock of pre-civil war plays was divided between the two newly created patent
theatre companies , Hamlet was the only Shakespearean favourite that Sir William Davenant 's Duke 's Company secured . It became the first of Shakespeare 's
plays to be presented with movable flats painted with generic scenery behind the proscenium arch of Lincoln 's Inn Fields Theatre . This new stage convention
highlighted the frequency with which Shakespeare shifts dramatic location , encouraging the recurrent criticism of his failure to maintain unity of place . In
the title role , Davenant cast Thomas Betterton , who continued to play the Dane until he was 74 . David Garrick at Drury Lane produced a version that adapted
Shakespeare heavily ; he declared : `` I had sworn I would not leave the stage till I had rescued that noble play from all the rubbish of the fifth act . I have
brought it forth without the grave - digger 's trick , Osrick , & the fencing match '' . The first actor known to have played Hamlet in North America is Lewis
Hallam . Jr.
</item 5>

<item 6>
Hamlet
( father ) Mary Arden ( mother ) Gilbert Shakespeare ( brother ) Joan Shakespeare ( sister ) Edmund Shakespeare ( brother ) Richard Shakespeare ( grandfather )
John Hall ( son - in - law ) Thomas Quiney ( son - in - law ) Thomas Nash ( grandson - in - law ) * Shakespeare and other authors Lost William Shakespeare 's
Hamlet Characters Hamlet Claudius Gertrude Ghost Polonius Laertes Ophelia Horatio Rosencrantz and Guildenstern Fortinbras The Gravediggers Yorick Soliloquies ``
To be , or not to be '' `` Mortal coil '' `` What a piece of work is a man '' `` Speak the speech '' Words and phrases `` The lady doth protest too much ,
methinks '' `` Thy name is '' `` Primrose path '' Terminology Dumbshow Induction Quiddity Substitution Sources Criticism Legend of Hamlet The Spanish Tragedy Ur
- Hamlet Critical approaches Bibliographies Horwendill Saxo Grammaticus House of Gonzaga Damon and Pythias Influence Common phrases from Hamlet Hamlet in
popular culture References to Ophelia Language of flowers Human skull symbolism Performances Moscow Art Theatre ( 1911 -- 1912 )
</item 6>

<item 7>
Hamlet
William Shakespeare Plays Tragedies Antony and Cleopatra Coriolanus Hamlet Julius Caesar King Lear Macbeth Othello Romeo and Juliet Timon of Athens Titus
Andronicus Troilus and Cressida Comedies All 's Well That Ends Well As You Like It The Comedy of Errors Cymbeline Love 's Labour 's Lost Measure for Measure The
Merchant of Venice The Merry Wives of Windsor A Midsummer Night 's Dream Much Ado About Nothing Pericles , Prince of Tyre * The Taming of the Shrew The Tempest
Twelfth Night The Two Gentlemen of Verona The Two Noble Kinsmen * The Winter 's Tale Histories King John Edward III * Richard II Henry IV Henry V Henry VI 1 *
Richard III Henry VIII * See also Problem plays Late romances Characters A -- K L -- Z Ghost character Chronology Performances Settings Quarto publications
First Folio Second Folio Poems Shakespeare 's sonnets comparison to Petrarch A Lover 's Complaint The Phoenix and the Turtle The Rape of Lucrece Venus and
Adonis Apocrypha Plays Arden of Faversham The Birth of Merlin Cardenio * Double Falsehood Edmund Ironside Fair Em Locrine The London
</item 7>

<item 8>
Hamlet
He is dethroned by the spectre of an actor , and we shall never be able to keep the usurper out of our dreams . Maurice Maeterlinck in La Jeune Belgique ( 1890
) . Shakespeare 's day to the Interregnum ( edit ) Shakespeare almost certainly wrote the role of Hamlet for Richard Burbage . He was the chief tragedian of the
Lord Chamberlain 's Men , with a capacious memory for lines and a wide emotional range . Judging by the number of reprints , Hamlet appears to have been
Shakespeare 's fourth most popular play during his lifetime -- only Henry IV Part 1 , Richard III and Pericles eclipsed it . Shakespeare provides no clear
indication of when his play is set ; however , as Elizabethan actors performed at the Globe in contemporary dress on minimal sets , this would not have affected
the staging . Firm evidence for specific early performances of the play is scant .
</item 8>

<item 9>
Hamlet
The Birth of Merlin Cardenio * Double Falsehood Edmund Ironside Fair Em Locrine The London Prodigal Love 's Labour 's Won The Merry Devil of Edmonton Mucedorus
The Puritan The Second Maiden 's Tragedy Sejanus His Fall Sir John Oldcastle Sir Thomas More * The Spanish Tragedy Thomas Lord Cromwell Thomas of Woodstock
Vortigern and Rowena A Yorkshire Tragedy Poems The Passionate Pilgrim To the Queen Life and works Birthplace and childhood home Bibliography Complete Works of
William Shakespeare Translations Early editions Editors English Renaissance theatre Globe Theatre Handwriting Lord Chamberlain 's Men / King 's Men The Theatre
Curtain Theatre New Place Portraits Religious views Sexuality Spelling of his name Stratford - upon - Avon Style Legacy Attribution studies Authorship question
Influence Memorials Screen adaptations Shakespeare Birthplace Trust Folger Shakespeare Library Shakespeare Quarterly Shakespeare 's Globe ( replica ) Bardolatry
Titles of works taken from Shakespeare Family Anne Hathaway ( wife ) Susanna Hall ( daughter ) Hamnet Shakespeare ( son ) Judith Quiney ( daughter ) Elizabeth
Barnard ( granddaughter ) John Shakespeare ( father ) Mary Arden ( mother ) Gilbert Shakespeare ( brother ) Joan Shakespeare ( sister ) Edmund
</item 9>

<item 10>
Hamlet
Hamlet is not among them , suggesting that it had not yet been written . As Hamlet was very popular , Bernard Lott , the series editor of New Swan , believes it
`` unlikely that he ( Meres ) would have overlooked ... so significant a piece '' . The phrase `` little eyases '' in the First Folio ( F1 ) may allude to the
Children of the Chapel , whose popularity in London forced the Globe company into provincial touring . This became known as the War of the Theatres , and
supports a 1601 dating . Katherine Duncan - Jones accepts a 1600 -- 1 attribution for the date Hamlet was written , but notes that the Lord Chamberlain 's Men ,
playing Hamlet in the 3000 - capacity Globe , were unlikely to be put to any disadvantage by an audience of `` barely one hundred '' for the Children of the
Chapel 's equivalent play , Antonio 's Revenge ; she believes that Shakespeare , confident in the superiority of his own work , was making a playful and
charitable allusion to his friend John Marston 's very similar piece .
</item 10>
=== END ITEM id=ecf8f5a7e1385f87 ===

=== ITEM id=11f026e8a5d84081 ===
Gold sentences and facts:
Sentence 0: Angular cheilitis can be caused by infection , irritation , or allergies . Infections include by fungi such as Candida albicans and bacteria such as Staph . aureus . Irritants include poorly fitting dentures , licking the lips or drooling , mouth breathing resulting in a dry mouth , sun exposure , overclosure of the mouth , smoking , and minor trauma . Allergies may include substances like toothpaste , makeup , and food . Often a number of factors are involved . Other factors may include poor nutrition or poor immune function . Diagnosis may be helped by testing for infections and patch testing for allergies .
  Fact 0.0: Angular cheilitis can be caused by infection.
  Fact 0.1: Angular cheilitis can be caused by irritation.
  Fact 0.2: Angular cheilitis can be caused by allergies.
  Fact 0.3: Infections that cause angular cheilitis include infection by fungi such as Candida albicans.
  Fact 0.4: Infections that cause angular cheilitis include infection by bacteria such as Staph. aureus.
  Fact 0.5: Irritants that cause angular cheilitis include poorly fitting dentures.
  Fact 0.6: Irritants that cause angular cheilitis include licking the lips or drooling.
  Fact 0.7: Irritants that cause angular cheilitis include mouth breathing resulting in a dry mouth.
  Fact 0.8: Irritants that cause angular cheilitis include sun exposure.
  Fact 0.9: Irritants that cause angular cheilitis include overclosure of the mouth.
  Fact 0.10: Irritants that cause angular cheilitis include smoking.
  Fact 0.11: Irritants that cause angular cheilitis include minor trauma.
  Fact 0.12: Allergies that cause angular cheilitis may include substances like toothpaste.
  Fact 0.13: Allergies that cause angular cheilitis may include substances like makeup.
  Fact 0.14: Allergies that cause angular cheilitis may include substances like food.
  Fact 0.15: Often a number of factors are involved in angular cheilitis.
  Fact 0.16: Other factors may include poor nutrition.
  Fact 0.17: Other factors may include poor immune function.
  Fact 0.18: Diagnosis of angular cheilitis may be helped by testing for infections.
  Fact 0.19: Diagnosis of angular cheilitis may be helped by patch testing for allergies.

Retrieved items:
<item 1>
Angular cheilitis
Angular cheilitis - wikipedia Angular cheilitis Jump to : navigation , search Angular cheilitis Synonyms Rhagades , perlèche , cheilosis , angular cheilosis ,
commissural cheilitis , angular stomatitis Bilateral angular cheilitis in an elderly individual with false teeth , iron deficiency anemia and dry mouth
Pronunciation / kaɪˈlaɪtɪs / Specialty Dermatology Symptoms Redness , skin breakdown and crusting at the corner of the mouth Usual onset Children , 30s to 60s
Duration Days to years Causes Infection , irritation , allergies Treatment Based on cause , barrier cream Frequency 0.7 % of the population Angular cheilitis (
AC ) , is inflammation of one or both corners of the mouth . Often the corners are red with skin breakdown and crusting . It can also be itchy or painful . The
condition can last for days to years . Angular cheilitis is a type of cheilitis ( inflammation of the lips ) . Angular cheilitis can be caused by infection ,
irritation , or allergies . Infections include by fungi such as Candida albicans and bacteria such as Staph .
</item 1>

<item 2>
Angular cheilitis
Infections include by fungi such as Candida albicans and bacteria such as Staph . aureus . Irritants include poorly fitting dentures , licking the lips or
drooling , mouth breathing resulting in a dry mouth , sun exposure , overclosure of the mouth , smoking , and minor trauma . Allergies may include substances
like toothpaste , makeup , and food . Often a number of factors are involved . Other factors may include poor nutrition or poor immune function . Diagnosis may
be helped by testing for infections and patch testing for allergies . Treatment for angular cheilitis is typically based on the underlying causes along with the
use of a barrier cream . Frequently an antifungal and antibacterial cream is also tried . Angular cheilitis is a fairly common problem , with estimates that it
affects 0.7 % of the population . It occurs most often in the 30s to 60s , although is also relatively common in children . In the developing world , iron and
vitamin deficiencies are a common cause .
</item 2>

<item 3>
Angular cheilitis
The corner of the mouth is normally exposed to saliva more than any other part of the lips . Reduced lower facial height ( vertical dimension or facial support
) is usually caused by edentulism ( tooth loss ) , or wearing worn down , old dentures or ones which are not designed optimally . This results in overclosure of
the mandible ( collapse of the jaws ) , which extenuates the angular skin folds at the corners of the mouth , in effect creating an intertriginous skin crease .
The tendency of saliva to pool in these areas is increased , constantly wetting the area , which may cause tissue maceration and favors the development of a
yeast infection . As such , angular cheilitis is more commonly seen in edentulous people ( people without any teeth ) . It is by contrast uncommon in persons
who retain their natural teeth . Angular cheilitis is also commonly seen in denture wearers . Angular cheilitis is present in about 30 % of people with denture
- related stomatitis .
</item 3>

<item 4>
Angular cheilitis
Examples include cocaine , methamphetamines , heroin , and hallucinogens . Allergic contact dermatitis ( edit ) See also : Allergic contact cheilitis , Allergic
contact dermatitis , and Allergic contact stomatitis Patch test Allergic reactions may account for about 25 -- 34 % of cases of generalized cheilitis ( i.e. ,
inflammation not confined to the angles of the mouth ) . It is unknown how frequently allergic reactions are responsible for cases of angular cheilitis , but
any substance capable of causing generalized allergic cheilitis may present involving the corners of the mouth alone . Examples of potential allergens include
substances that may be present in some types of lipstick , toothpaste , acne products , cosmetics , chewing gum , mouthwash , foods , dental appliances , and
materials from dentures or mercury containing amalgam fillings . It is usually impossible to tell the difference between irritant contact dermatitis and
allergic contact dermatitis without a patch test . Diagnosis ( edit ) Photographic Comparison of : 1 ) a Canker Sore - inside the mouth , 2 ) Herpes , 3 )
Angular Cheilitis and 4 ) Chapped Lips .
</item 4>

<item 5>
Angular cheilitis
In the developing world , iron and vitamin deficiencies are a common cause . Contents ( hide ) 1 Signs and symptoms 2 Causes 2.1 Infection 2.2 Irritation
contact dermatitis 2.3 Nutritional deficiencies 2.4 Systemic disorders 2.5 Drugs 2.6 Allergic contact dermatitis 3 Diagnosis 3.1 Classification 4 Management 5
Prognosis 6 Epidemiology 7 References 8 External links Signs and symptoms ( edit ) Angular cheilitis -- a fissure running in the corner of the mouth with
reddened , irritated facial skin adjacent . A fairly mild case of angular cheilitis extending onto the facial skin in a young person ( affected area is within
the black oval ) . Angular cheilitis is a fairly non specific term which describes the presence of an inflammatory lesion in a particular anatomic site ( i.e.
the corner of the mouth ) . As there are different possible causes and contributing factors from one person to the next , the appearance of the lesion is
somewhat variable . The lesions are more commonly symmetrically present on both sides of the mouth , but sometimes only one side may be affected .
</item 5>

<item 6>
Angular cheilitis
The term `` cheilocandidiasis '' describes exfoliative ( flaking ) lesions of the lips and the skin around the lips , and is caused by a superficial candidal
infection due to chronic lip licking . Less severe cases occur during cold , dry weather , and is a form of chapped lips . Individuals may lick their lips in an
attempt to provide a temporary moment of relief , only serving to worsen the condition . The sunscreen in some types of lip balm degrades over time into an
irritant . Using expired lipbalm can initiate mild angular cheilitis , and when the person applies more lipbalm to alleviate the cracking , it only aggravates
it . Because of the delayed onset of contact dermatitis and the recovery period lasting days to weeks , people typically do not make the connection between the
causative agent and the symptoms . Nutritional deficiencies ( edit ) Several different nutritional deficiency states of vitamins or minerals have been linked to
AC . It is thought that in about 25 % of people with AC , iron deficiency or deficiency of B vitamins are involved . Nutritional deficiencies may be a more
common cause of AC in Third World countries .
</item 6>

<item 7>
Angular cheilitis
It is thought that reduced vertical dimension of the lower face may be a contributing factor in up to 11 % of elderly persons with angular cheilitis and in up
to 18 % of denture wearers who have angular cheilitis . Reduced vertical dimension can also be caused by tooth migration , wearing orthodontic appliances , and
elastic tissue damage caused by ultraviolet light exposure and smoking . Habits or conditions that keep the corners of the mouth moist might include chronic lip
licking , thumb sucking ( or sucking on other objects such as pens , pipes , lollipops ) , dental cleaning ( e.g. flossing ) , chewing gum , hypersalivation ,
drooling and mouth breathing . Some consider habitual lip licking or picking to be a form of nervous tic , and do not consider this to be true angular cheilitis
, instead calling it perlèche ( derived from the French word pourlècher meaning `` to lick one 's lips '' ) , or `` factitious cheilitis '' is applied to this
habit .
</item 7>

<item 8>
Angular cheilitis
In chronic angular cheilitis , there may be suppuration ( pus formation ) , exfoliation ( scaling ) and formation of granulation tissue . Sometimes contributing
factors can be readily seen , such as loss of lower face height from poorly made or worn dentures , which results in mandibular overclosure ( `` collapse of
jaws '' ) . If there is a nutritional deficiency underlying the condition , various other signs and symptoms such as glossitis ( swollen tongue ) may be present
. In people with angular cheilitis who wear dentures , often there may be erythematous mucosa underneath the denture ( normally the upper denture ) , an
appearance consistent with denture - related stomatitis . Typically the lesions give symptoms of soreness , pain , pruritus ( itching ) or burning or a raw
feeling . Causes ( edit ) Angular cheilitis is thought to be multifactorial disorder of infectious origin , with many local and systemic predisposing factors .
The sores in angular cheilitis are often infected with fungi ( yeasts ) , bacteria , or a combination thereof ; this may represent a secondary , opportunistic
infection by these pathogens .
</item 8>

<item 9>
Angular cheilitis
A cold sore at the corner of the mouth behaves similarly to elsewhere on the lips , and follows a pattern of vesicle ( blister ) formation followed by rupture
leaving a crusted sore which resolves in about 7 -- 10 days , and recurs in the same spot periodically , especially during periods of stress . Rather than
utilizing antifungal creams , angular herpes simplex is treated in the same way as a cold sore , with topical antiviral drugs such as aciclovir . Irritation
contact dermatitis ( edit ) A famous sketch by Leonardo da Vinci in preparation to depict the face of Judas Iscariot in The Last supper . The subject shows
overclosure of the jaws and loss of facial support around the mouth . Pronounced skin folds extending from the corner of the mouth . Main article : Irritant
contact dermatitis 22 % of cases of angular cheilitis are due to irritants . Saliva contains digestive enzymes , which may have a degree of digestive action on
tissues if they are left in contact . The corner of the mouth is normally exposed to saliva more than any other part of the lips .
</item 9>

<item 10>
Angular cheilitis
Identification of the underlying cause is essential for treating chronic cases . The lesions may resolve when the underlying disease is treated , e.g. with a
course of oral iron or B vitamin supplements . Patch testing is recommended by some in cases which are resistant to treatment and where allergic contact
dermatitis is suspected . Prognosis ( edit ) Most cases of angular cheilitis respond quickly when antifungal treatment is used . In more long standing cases ,
the severity of the condition often follows a relapsing and remitting course over time . The condition can be difficult to treat and can be prolonged .
Epidemiology ( edit ) AC is a relatively common condition , accounting for between 0.7 -- 3.8 % of oral mucosal lesions in adults and between 0.2 -- 15.1 % in
children , though overall it occurs most commonly in adults in the third to sixth decades of life . It occurs worldwide , and both males and females are
affected . Angular cheilitis is the most common presentation of fungal and bacterial infections of the lips . References ( edit ) Jump up ^ Pindborg , Jens
Jørgen ( 1973 ) .
</item 10>
=== END ITEM id=11f026e8a5d84081 ===
