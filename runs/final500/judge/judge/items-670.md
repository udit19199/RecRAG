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

=== ITEM id=3436ef039558d827 ===
Gold sentences and facts:
Sentence 0: Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C. , bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming the longest serving moderator in the program 's history .
  Fact 0.0: Network officials were concerned for the show's future.
  Fact 0.1: Network officials turned to Tim Russert.
  Fact 0.2: Tim Russert was the network's Washington, D.C., bureau chief.
  Fact 0.3: Tim Russert took over as moderator of Meet the Press on December 8, 1991.
  Fact 0.4: Tim Russert remained with Meet the Press until his death on June 13, 2008.
  Fact 0.5: Tim Russert became the longest serving moderator in the history of Meet the Press.
Sentence 1: Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966 -- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley 1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present
  Fact 1.0: Martha Rountree was moderator from 1947 to 1953.
  Fact 1.1: Ned Brooks was moderator from 1953 to 1965.
  Fact 1.2: Lawrence E. Spivak was moderator from 1966 to 1975.
  Fact 1.3: Bill Monroe was moderator from 1975 to 1984.
  Fact 1.4: Roger Mudd and Marvin Kalb were co-moderators from 1984 to 1985.
  Fact 1.5: Marvin Kalb was moderator from 1985 to 1987.
  Fact 1.6: Chris Wallace was moderator from 1987 to 1988.
  Fact 1.7: Garrick Utley was moderator from 1989 to 1991.
  Fact 1.8: Tim Russert was moderator from 1991 to 2008.
  Fact 1.9: Tom Brokaw was moderator in 2008.
  Fact 1.10: David Gregory was moderator from 2008 to 2014.
  Fact 1.11: Chuck Todd has been moderator since 2014.

Retrieved items:
<item 1>
Meet the Press
Camera setup Multi-camera Running time 30 minutes ( 1947 -- 1992 ) 52 minutes ( 1992 -- present ) Production company ( s ) NBC News Productions Release Original
network NBC Picture format 480i ( 4 : 3 SDTV ) ( 1947 -- 2010 ) 720p ( 4 : 3 SDTV ) ( 1996 -- 2010 ) 1080i ( 16 : 9 HDTV ) ( 2010 -- present ) Original release
November 6 , 1947 ( 1947 - 11 - 06 ) -- present External links Website www.nbcnews.com/meet-the-press Meet the Press is a weekly American television news /
interview program broadcast on NBC . It is the longest - running program in television history , though the current format bears little resemblance to the debut
episode on November 6 , 1947 . Meet the Press specializes in interviews with leaders in Washington , D.C. , across the country and even the world on issues of
politics , economics , foreign policy and other public affairs , along with panel discussions that provide opinions and analysis . It originates from NBC 's
bureau in Washington , D.C.
</item 1>

<item 2>
Meet the Press
This section , in the next two paragraphs , needs additional citations for verification . Please help improve this article by adding citations to reliable
sources . Unsourced material may be challenged and removed . ( September 2014 ) ( Learn how and when to remove this template message ) Meet the Press was
originally presented as a 30 - minute press conference with a single guest and a panel of questioners . Its first guest was James Farley , who served as
Postmaster General , Democratic National Committee chairman and campaign manager to Franklin Delano Roosevelt under the first two terms of the New Deal
Administration . Creator Rountree served as its first host , the program 's only female moderator to date . She stepped down on November 1 , 1953 , and was
succeeded by Ned Brooks , who remained as moderator until his retirement on December 26 , 1965 . Spivak became the moderator on January 1 , 1966 , moving up
from his role as a permanent panelist . He retired on November 9 , 1975 , on a special one - hour edition that featured , for the first time , a sitting
president , Gerald Ford , as guest .
</item 2>

<item 3>
Meet the Press
The next week , Bill Monroe , previously a weekly panelist like Spivak had been years before , took over as moderator and stayed until June 2 , 1984 . For the
next seven and a half years , the program then went through a series of hosts as it struggled in the ratings against ABC 's This Week with David Brinkley .
Roger Mudd and Marvin Kalb , as co-moderators , followed Monroe for a year , followed by Chris Wallace ( who would later to go on to a much longer run as host
of the rival program Fox News Sunday ) from 1987 to 1988 . Garrick Utley , then hosting Weekend Today , concurrently hosted Meet the Press from 1989 through
December 1 , 1991 . All this occurred despite the increasing ratings of NBC News ' other programs ( and those of the network generally ) during that period .
The program originally aired at noon Eastern Time every Sunday , before moving to a 9 : 00 a.m. slot by the early 1990s . Under Russert ( edit ) Meet the Press
logo used from September 10 , 1995 to June 2008 .
</item 3>

<item 4>
Meet the Press
On May 2 , 2010 , Meet the Press became the last NBC News program to convert to high definition , and unveiled a new set consisting of large video screens
mostly used to display Washington scenery , satellite interview subjects and moderator and subject talking points , along with graphics produced for the format
. Moderators ( edit ) The following is the list of moderators for Meet the Press : Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966
-- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley
1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present On - remote broadcasts ( outside D.C. studios ) (
edit ) Date Location Event / Reason July 17 , 1988 Atlanta , Georgia 1988 Democratic National Convention August 14 , 1988 New Orleans ,
</item 4>

<item 5>
Meet the Press
NBC News . Retrieved August 14 , 2014 . Jump up ^ David Paul Kuhn ( 2008 - 06 - 13 ) . `` Memorable Tim Russert moments '' . Politico . Retrieved 2008 - 06 - 14
. Jump up ^ `` 60 Years Ago in News History : America Meets the Press '' . Newseum . Archived from the original on November 17 , 2008 . Jump up ^ `` American
Mercury Sold to L.E. Spivak '' . The New York Times . January 23 , 1935 . Retrieved 2017 - 08 - 02 . Jump up ^ `` Fast facts about the longest - running program
in TV history '' . MSNBC.com . Jump up ^ `` In the Hot Seat '' . The Washington Post . May 23 , 2004 . Retrieved May 12 , 2010 . Jump up ^ `` Tim Russert 's
Commencement Address -- CUA Office of Public Affairs '' . Jump up ^ `` Tim Russert hits ratings milestone '' . USA Today .
</item 5>

<item 6>
Meet the Press
studios ) 3.2 Notable guests and events 4 Distribution 5 See also 6 References and footnotes 7 External links Format ( edit ) The program 's format consists of
an extended one - on - one interview with the host , and is sometimes followed by a roundtable discussion or one - on - two interview with figures in
adversarial positions , either Congressional members from opposite sides of the aisle or political commentators . Originally a half - hour program for most of
its history , the show expanded to 60 minutes starting with the September 20 , 1992 , broadcast . The program also features in - depth examinations of facts
behind political and general news stories ( particularly as part of a segment called the `` Data Download '' , introduced after Chuck Todd assumed duties as
moderator , which is conducted on a touchscreen within the main set . History ( edit ) Meet the Press set , November 1975 . On this broadcast , a sitting
American president ( Gerald Ford ) was , for the first time , a guest on a live television network news program .
</item 6>

<item 7>
Meet the Press
First guest : James A. Farley , the former Postmaster General of the United States and former Democratic National Committee Chair . Whittaker Chambers states
Alger Hiss was a communist on August 27 , 1948 , radio broadcast , which leads to libel suit from Hiss , the Pumpkin Papers , and Justice 's indictment of Hiss
by December 1948 . First female guest : Elizabeth Bentley , a courier for a Communist spy ring , on September 12 , 1948 . An interview with Fidel Castro aired
April 19 , 1959 . Every U.S. President since John F. Kennedy has appeared on Meet the Press , although not necessarily during his presidency . Jimmy Carter used
his appearance on January 20 , 1980 , to announce the United States ' boycott of the 1980 Summer Olympics . Ronald Reagan appeared seven times before being
elected president , but did not appear during his presidency . Bill Clinton was the guest for the 50th anniversary broadcast on November 9 , 1997 . The February
8 , 2004 , interview with George W. Bush was conducted in the Oval Office at the White House .
</item 7>

<item 8>
Meet the Press
Russert interviews General Peter Pace in 2006 . Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C.
, bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming
the longest serving moderator in the program 's history . Under Russert , the program was expanded to one hour and became less of a televised press conference ,
focusing more on Russert 's questions and comments ; Russert also engaged in longer in - depth interviews and hosted panels of experts to discuss the topics
featured in that week 's broadcast . Russert signed off each edition by saying , `` That 's all for today . We 'll be back next week . If it 's Sunday , it 's
Meet the Press . '' During the professional football season , Russert , a native of Buffalo , New York , and an avid fan of the Buffalo Bills , sometimes added
, `` Go Bills ! , '' and occasionally would ask panelists , `` How ' bout those Sabres ? ''
</item 8>

<item 9>
Meet the Press
It originates from NBC 's bureau in Washington , D.C. The longevity of Meet the Press can be illustrated in consideration of the fact that the program debuted
during what was only the second official `` network television season '' for American television . One historical landmark of the program is that it was the
first on which a sitting U.S. President , Gerald Ford , appeared on a live television network news program , which occurred on the November 9 , 1975 broadcast .
The program has been hosted by 12 different moderators to date , beginning with creator Martha Rountree . The show 's moderator since 2014 is Chuck Todd , who
also serves as political director for NBC News . Currently , the hour - long program airs in most markets on Sundays at 9 : 00 a.m. live in the Eastern Time
Zone and on tape delay elsewhere . Meet the Press is also occasionally pre-empted due to network coverage of sports events held outside of the U.S. The program
is also rebroadcast on Sundays at 2 : 00 p.m. , and Mondays at 2 : 00 a.m. and sometimes 4 : 00 a.m.
</item 9>

<item 10>
Meet the Press
The following week on August 17 , he moderated the entire broadcast . It was also reported on December 1 , 2008 , that the December 7 broadcast would be Brokaw
's last , with Gregory becoming the new permanent host the following Sunday . Under Gregory ( edit ) David Gregory began his tenure as moderator on December 14
, 2008 . Four days after Gregory 's first regular broadcast , on December 18 , NBC News political director Chuck Todd was named contributing editor of Meet the
Press . Throughout Gregory 's tenure as moderator , Meet the Press experienced significant ratings declines . In the final three months of 2013 , the program
placed third among the Sunday morning talk shows in total viewership , behind CBS 's Face the Nation and ABC 's This Week for the first time since 1992 ; it
also experienced the lowest ratings in the show 's entire history among the key 25 - to - 54 age viewing demographic during this period . NBC management became
uncertain as to the future direction of the program .
</item 10>
=== END ITEM id=3436ef039558d827 ===

=== ITEM id=680a5873d9a9eda4 ===
Gold sentences and facts:
Sentence 0: The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of the Mahanadi Delta .
  Fact 0.0: The Eastern Coastal Plains refers to a wide stretch of landmass of India.
  Fact 0.1: The Eastern Coastal Plains lie between the Eastern Ghats and the Bay of Bengal.
  Fact 0.2: The Eastern Coastal Plains are wider and more leveled than the western coastal plains.
  Fact 0.3: The Eastern Coastal Plains stretch from Tamil Nadu in the south to West Bengal in the north.
  Fact 0.4: The Eastern Coastal Plains stretch through Andhra Pradesh and Odisha.
  Fact 0.5: Chilka Lake is a brackish water lake along the eastern coastal plain.
  Fact 0.6: Chilka Lake lies in the state of Odisha.
  Fact 0.7: Chilka Lake stretches to the south of the Mahanadi Delta.
Sentence 1: It is locally known as Utkal Plains in the Northern part between Cossye and Rushikulya Rivers , Northern Circars in the Central part between Rushikulya and Krishna Rivers and , as Coromandel Coast in the Southern part from the south of river Krishna till the Southern tip of Mainland India at Cape Comorin where it merges with the Western Coastal Plains .
  Fact 1.0: The Eastern Coastal Plains are locally known as Utkal Plains in the northern part between the Cossye and Rushikulya Rivers.
  Fact 1.1: The Eastern Coastal Plains are known as Northern Circars in the central part between the Rushikulya and Krishna Rivers.
  Fact 1.2: The Eastern Coastal Plains are known as Coromandel Coast in the southern part from the south of the river Krishna to the southern tip of mainland India at Cape Comorin.
  Fact 1.3: The Coromandel Coast merges with the Western Coastal Plains at Cape Comorin.

Retrieved items:
<item 1>
Eastern coastal plains
pp. 20 -- 21 . ISBN 8171725163 . Geography of India Climate Climate Climatic regions Geology Fossil Parks Geology of India Indian Plate Stones Landforms Beaches
Canals Desert Extreme points Glaciers Islands Lakes Mountains Rivers Valleys Volcanoes Waterfalls Plains Indo - Gangetic Eastern Coastal Western Coastal Regions
North India Northeast India East India South India West India Central India Subdivisions Autonomous administrative divisions States and territories Districts
Cities Towns Municipalities Environment Biogeographic classification Ecoregions Fauna Flora Forests Issues Protected Areas Wildlife This Indian location article
is a stub . You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Eastern_coastal_plains&oldid=812520937 ''
Categories : Plains of India Coasts of India India geography stubs Hidden categories : Articles needing additional references from May 2015 All articles needing
additional references India articles missing geocoordinate data All articles needing coordinates All stub articles Talk About Wikipedia हिन्दी Português தமிழ்
Edit links This page was last edited on 28 November 2017 , at 08 : 32 .
</item 1>

<item 2>
Ring of Fire
Plateau Coromandel Coast Konkan False Divi Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen
Glacier Bay of Bengal Gulf of Khambhat Gulf of Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands
Andaman Islands Nicobar Islands Maldive Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater
Sunda Islands Lesser Sunda Islands Indonesian Archipelago Timor New Guinea Bonis Peninsula Papuan Peninsula Huon Peninsula Huon Gulf Bird 's Head Peninsula
Gazelle Peninsula Philippine Archipelago Luzon Visayas Mindanao Leyte Gulf Gulf of Thailand East Indies Nanyang Alpide belt Asia - Pacific Tropical Asia Ring of
Fire Regions of Europe North Nordic Northwestern Scandinavia Scandinavian Peninsula Fennoscandia Baltoscandia Sápmi West Nordic Baltic Baltic Sea Gulf of
Bothnia Gulf of Finland Iceland Faroe Islands East Danubian countries Prussia Galicia Volhynia Donbass Sloboda Ukraine Sambia Peninsula Amber Coast Curonian
Spit Izyum
</item 2>

<item 3>
Dead Sea
Kur - Araz Lowland Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt South Greater India Indian subcontinent Himalayas Hindu
Kush Western Ghats Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir Kashmir Valley Pir Panjal Range Thar Desert Indus Valley
Indus River Delta Indus Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic
plains moist deciduous forests Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau
Coromandel Coast Konkan False Divi Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen Glacier Bay
of Bengal Gulf of Khambhat Gulf of Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands Andaman
Islands Nicobar Islands Maldive Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater
</item 3>

<item 4>
Graph query row:
plain: Eastern Coastal Plains
relationship: LOCATED_IN
related: India
related_kind: country
detail: None
</item 4>

<item 5>
Sub-Saharan Africa
Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic plains moist deciduous forests
Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau Coromandel Coast Konkan False Divi
Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen Glacier Bay of Bengal Gulf of Khambhat Gulf of
Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands Andaman Islands Nicobar Islands Maldive
Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater Sunda Islands Lesser Sunda Islands
Indonesian Archipelago Timor New Guinea Bonis Peninsula Papuan Peninsula Huon Peninsula Huon Gulf Bird 's Head Peninsula Gazelle Peninsula Philippine
Archipelago Luzon Visayas Mindanao Leyte Gulf Gulf of Thailand East Indies Nanyang Alpide belt Asia - Pacific Tropical Asia Ring of Fire Regions of Europe North
Nordic Northwestern Scandinavia Scandinavian Peninsula
</item 5>

<item 6>
Eastern coastal plains
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 6>

<item 7>
Eastern coastal plains
Eastern coastal plains - Wikipedia Eastern coastal plains Jump to : navigation , search This article relies largely or entirely on a single source . Relevant
discussion may be found on the talk page . Please help improve this article by introducing citations to additional sources . ( May 2015 ) View of Fields at
Biccavolu , Eastern coastal plains , Andhra Pradesh The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats
and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through
Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of
the Mahanadi Delta . Deltas of many of India 's rivers form a major portion of these plains . The Mahanadi , Godavari , Kaveri and Krishna rivers drain these
plains . The region receives both the Northeast & Southwest monsoon rains with its annual rainfall averaging between 1,000 and 3,000 mm ( 39 and 118 in ) .
</item 7>

<item 8>
Sub-Saharan Africa
Arabian Peninsula Najd Hejaz Tihamah Eastern Arabia South Arabia Hadhramaut Arabian Peninsula coastal fog desert Tigris -- Euphrates Mesopotamia Upper
Mesopotamia Lower Mesopotamia Sawad Nineveh plains Akkad ( region ) Babylonia Canaan Aram Eber - Nari Suhum Eastern Mediterranean Mashriq Kurdistan Levant
Southern Levant Transjordan Jordan Rift Valley Israel Levantine Sea Golan Heights Hula Valley Galilee Gilead Judea Samaria Arabah Anti-Lebanon Mountains Sinai
Peninsula Arabian Desert Syrian Desert Fertile Crescent Azerbaijan Syria Palestine Iranian Plateau Armenian Highlands Caucasus Caucasus Mountains Greater
Caucasus Lesser Caucasus North Caucasus South Caucasus Kur - Araz Lowland Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt
South Greater India Indian subcontinent Himalayas Hindu Kush Western Ghats Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir
Kashmir Valley Pir Panjal Range Thar Desert Indus Valley Indus River Delta Indus Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal
Plains Meghalaya subtropical forests MENASA Lower
</item 8>

<item 9>
Eastern coastal plains
The width of the plains varies between 100 and 130 km ( 62 to 80 miles ) . It is locally known as Utkal Plains in the Northern part between Cossye and
Rushikulya Rivers , Northern Circars in the Central part between Rushikulya and Krishna Rivers and , as Coromandel Coast in the Southern part from the south of
river Krishna till the Southern tip of Mainland India at Cape Comorin where it merges with the Western Coastal Plains . See also ( edit ) Western Coastal Plains
References ( edit ) Jump up ^ Sr. Bimcy ; Sr. Sisily ; Charlotte . Bibliographic information . Scholar Publishing House . pp. 20 -- 21 . ISBN 8171725163 .
</item 9>

<item 10>
Ring of Fire
Aram Eber - Nari Suhum Eastern Mediterranean Mashriq Kurdistan Levant Southern Levant Transjordan Jordan Rift Valley Levantine Sea Golan Heights Hula Valley
Gaza Strip West Bank Galilee Gilead Judea Samaria Arabah Anti-Lebanon Mountains Sinai Peninsula Arabian Desert Syrian Desert Fertile Crescent Azerbaijan Syria
Palestine Iranian Plateau Armenian Highlands Caucasus Caucasus Mountains Greater Caucasus Lesser Caucasus North Caucasus South Caucasus Kur - Araz Lowland
Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt South Greater India Indian subcontinent Himalayas Hindu Kush Western Ghats
Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir Kashmir Valley Pir Panjal Range Thar Desert Indus Valley Indus River Delta Indus
Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic plains moist deciduous
forests Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau Coromandel Coast Konkan False
Divi Point Hindi Belt Ladakh Aksai Chin
</item 10>
=== END ITEM id=680a5873d9a9eda4 ===

=== ITEM id=66f3f6ae9e39f435 ===
Gold sentences and facts:
Sentence 0: The uvea is the vascular middle layer of the eye . It is traditionally divided into three areas , from front to back , the :
  Fact 0.0: The uvea is the vascular middle layer of the eye.
  Fact 0.1: The uvea is traditionally divided into three areas.
  Fact 0.2: The three areas of the uvea are divided from front to back.

Retrieved items:
<item 1>
Retina
The optic disc is at right , and the macula lutea is near the centre . The retina is stratified into distinct layers , each containing specific cell types or
cellular compartments that have metabolisms with different nutritional requirements . To satisfy these requirements , the ophthalmic artery bifurcates and
supplies the retina via two distinct vascular networks : the choroidal network , which supplies the choroid and the outer retina , and the retinal network ,
which supplies the retina 's inner layer . Circulatory mechanisms ( edit ) At first glance , one may think that the vertebrate retina is `` wired wrongly '' or
`` badly designed '' ; but in fact , the retina could not function if it were not inverted . The photoreceptor layer must be embedded in the retinal pigment
epithelium ( RPE ) , which performs at least seven vital functions , one of the most obvious being to supply oxygen and other necessary nutrients needed for the
photoreceptors to function . These nutrients include glucose , fatty acids , and retinal .
</item 1>

<item 2>
Retina
In section , the retina is no more than 0.5 mm thick . It has three layers of nerve cells and two of synapses , including the unique ribbon synapse . The optic
nerve carries the ganglion cell axons to the brain , and the blood vessels that supply the retina . The ganglion cells lie innermost in the eye while the
photoreceptive cells lie beyond . Because of this counter-intuitive arrangement , light must first pass through and around the ganglion cells and through the
thickness of the retina , ( including its capillary vessels , not shown ) before reaching the rods and cones . Light is absorbed by the retinal pigment
epithelium or the choroid ( both of which are opaque ) . The white blood cells in the capillaries in front of the photoreceptors can be perceived as tiny bright
moving dots when looking into blue light . This is known as the blue field entoptic phenomenon ( or Scheerer 's phenomenon ) . Between the ganglion cell layer
and the rods and cones there are two layers of neuropils where synaptic contacts are made . The neuropil layers are the outer plexiform layer and the inner
plexiform layer .
</item 2>

<item 3>
Uvea
( The cornea has no adjacent blood vessels and is oxygenated by direct gas exchange with the environment . ) Light absorption : the uvea improves the contrast
of the retinal image by reducing reflected light within the eye ( analogous to the black paint inside a camera ) , and also absorbs outside light transmitted
through the sclera , which is not fully opaque . In addition some uveal regions have special functions of great importance , including secretion of the aqueous
humour by the ciliary processes , control of accommodation ( focus ) by the ciliary body , and optimisation of retinal illumination by the iris 's control over
the pupil . Many of these functions are under the control of the autonomic nervous system . Pharmacology ( edit ) The pupil provides a visible example of the
neural feedback control in the body . This is subserved by a balance between the antagonistic sympathetic and parasympathetic divisions of the autonomic nervous
system . Informal pharmacological experiments have been performed on the pupil for centuries , since the pupil is readily visible , and its size can be readily
altered by applying drugs -- even crude plant extracts -- to the cornea .
</item 3>

<item 4>
Retina
The layers and anatomical correlation are as follows : OCT scan of the macular area of a retina at 800 nm , axial resolution 3 μm From innermost to outermost ,
the layers identifiable by OCT are as follows : # OCT Layer / Conventional Label Anatomical Correlate Reflectivity on OCT Specific anatomical boundaries ?
Additional references Posterior cortical vitreous Posterior cortical vitreous Hyper - reflective Yes Preretinal space In eyes where the vitreous has fully or
partially detached from the retina , this is the space created between the posterior cortical vitreous face and the internal limiting membrane of the retina .
Hypo - reflective Internal limiting membrane ( ILM ) Formed by Müller cell endfeet ( unclear if it can be observed on OCT ) Hyper - reflective No Nerve fiber
layer ( NFL ) Ganglion cell axons travelling towards the optic nerve Ganglion cell layer ( GCL ) Ganglion cell bodies ( and some displaced amacrine cells ) Hypo
- reflective 5 Inner plexiform layer ( IPL ) Synapses between bipolar , amacrine and ganglion cells Hyper - reflective 6 Inner nuclear layer ( INL ) a )
Horizontal ,
</item 4>

<item 5>
Retina
Inner nuclear layer -- contains the nuclei and surrounding cell bodies ( perikarya ) of the amacrine cells , bipolar cells , and horizontal cells . Outer
plexiform layer -- projections of rods and cones ending in the rod spherule and cone pedicle , respectively . These make synapses with dendrites of bipolar
cells . In the macular region , this is known as the Fiber layer of Henle . Outer nuclear layer -- cell bodies of rods and cones . External limiting membrane --
layer that separates the inner segment portions of the photoreceptors from their cell nucleus . Layer of rods and cones -- layer of rod cells and cone cells .
Retinal pigment epithelium -- single layer of cuboidal cells ( with extrusions not shown in diagram ) . This is closest to the choroid . These layers can be
grouped into 4 main processing stages : photoreception ; transmission to bipolar cells ; transmission to ganglion cells , which also contain photoreceptors ,
the photosensitive ganglion cells ; and transmission along the optic nerve . At each synaptic stage there are also laterally connecting horizontal and amacrine
cells .
</item 5>

<item 6>
Uvea
Uvea - wikipedia Uvea Jump to : navigation , search For the Pacific island and chiefdom in Wallis and Futuna , see Wallis ( island ) and Uvea ( Wallis and
Futuna ) . For the Pacific commune and island in New Caledonia , see Ouvéa and Ouvéa Island . This article needs additional citations for verification . Please
help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( February 2014 ) ( Learn how and when to
remove this template message ) Uvea Horizontal section of the eyeball . The constituents of the uvea follow : iris labeled at top , ciliary body labeled at
upper right , and choroid labeled at center right . ) Details Identifiers Latin tunica vasculosa bulbi MeSH D014602 TA A15. 2.03. 001 FMA 58103 Anatomical
terminology ( edit on Wikidata ) The uvea ( / ˈjuːvɪə / ) ( Lat .
</item 6>

<item 7>
Retina
The optics of the eye create a focused two - dimentional image of the visual world on the retina , which translates that image into electrical neural impulses
to the brain to create visual perception , the retina serving much the same function as film or a CCD in a camera . The retina consists of several layers of
neurons interconnected by synapses . The neural retina refers to the three layers of neural cells ( photo receptor cells , bipolar cells , and ganglion cells )
within the retina , which in its entirety comprises ten distinct layers , including an outer layer of pigmented epithelial cells . The only neural cells that
are directly sensitive to light are the photoreceptor cells , which are of two types : rods and cones . Rods function mainly in dim light and provide black -
and - white vision while cones are responsible for the perception of colour . A third type of photoreceptor , the photosensitive ganglion cells , is important
for entrainment and reflexive responses to the brightness of light . Light striking the retina initiates a cascade of chemical and electrical events that
ultimately trigger nerve impulses that are sent to various visual centres of the brain through the fibres of the optic nerve .
</item 7>

<item 8>
Uvea
M cell , K cell , Muller glia Other Macula Perifoveal area Parafoveal area Fovea Foveal avascular zone Foveola Optic disc Optic cup Ora serrata Anatomical
regions of the eye Anterior segment Adnexa Fibrous tunic Anterior chamber Aqueous humour Iris Posterior chamber Ciliary body Lens Capsule of lens Zonule of Zinn
Posterior segment Vitreous chamber Vitreous body Retina Choroid Other Keratocytes Ocular immune system Optical coherence tomography Eye care professional Eye
disease Refractive error Accommodation Physiological Optics Visual perception Retrieved from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
Uvea
com ( hide ) Anatomy of the globe of the human eye Fibrous tunic ( outer ) Sclera Episcleral layer Schlemm 's canal Trabecular meshwork Cornea Limbus layers
Epithelium Bowman 's Stroma Dua 's layer Descemet 's Endothelium Uvea / vascular tunic ( middle ) Choroid Capillary lamina of choroid Bruch 's membrane Sattler
's layer Ciliary body Ciliary processes Ciliary muscle Pars plicata Pars plana Iris Stroma Pupil Iris dilator muscle Iris sphincter muscle Retina ( inner )
Layers Inner limiting membrane Nerve fiber layer Ganglion cell layer Inner plexiform layer Inner nuclear layer Outer plexiform layer Outer nuclear layer
External limiting membrane Layer of rods and cones Retinal pigment epithelium Cells Photoreceptor cells ( Cone cell , Rod cell ) → ( Horizontal cell ) → Bipolar
cell → ( Amacrine cell ) → Retina ganglion cell ( Midget cell , Parasol cell , Bistratified cell , Giant retina ganglion cells , Photosensitive ganglion cell )
→ Diencephalon : P cell , M cell , K cell ,
</item 9>

<item 10>
Uvea
uva , grape ) , also called the uveal layer , uveal coat , uveal tract , or vascular tunic , is the pigmented middle of the three concentric layers that make up
an eye . The name is possibly a reference to its reddish - blue or almost black colour , wrinkled appearance and grape - like size and shape when stripped
intact from a cadaveric eye . Its use as a technical term in anatomy and ophthalmology is relatively modern . Contents ( hide ) 1 Structure 1.1 Regions 2
Function 3 Pharmacology 4 Immunology 5 Clinical significance 6 References 7 External links Structure ( edit ) Regions ( edit ) The uvea is the vascular middle
layer of the eye . It is traditionally divided into three areas , from front to back , the : Iris Ciliary body Choroid Function ( edit ) The prime functions of
the uveal tract as a unit are : Nutrition and gas exchange : uveal vessels directly perfuse the ciliary body and iris , to support their metabolic needs , and
indirectly supply diffusible nutrients to the outer retina , sclera , and lens , which lack any intrinsic blood supply .
</item 10>
=== END ITEM id=66f3f6ae9e39f435 ===

=== ITEM id=7d025170afbe343f ===
Gold sentences and facts:
Sentence 0: The military convention between Bulgaria and the Central Powers laid down the general plan for its campaign against the Kingdom of Serbia . It severely limited the control of the Bulgarian High Command over the Bulgarian 1st Army , which was designated part of a combined German , Bulgarian and Austro - Hungarian force commanded by Field Marshal August von Mackensen . He had recently led the German and Austro - Hungarian armies in the highly successful and victorious Gorlice -- Tarnów Offensive of the Central Powers against the Russian army on the Eastern Front . His army group was created specifically to wage war against the Serbian army in the pre-1913 borders of the country ( `` Old Serbia '' ) , to defeat it wherever it found it and to open the land route between Hungary and Bulgaria . As commander , Mackensen acted independently and received his directives only from the German High Command . However , the field marshal 's orders to his Bulgarian forces had to be relayed to the commander of the 1st Army by the Bulgarian General Staff , which left room for the latter to intervene when needed . According to the convention , the Bulgarian commander - in - chief retained full and direct control over the Bulgarian 2nd Army and its operations in Vardar Macedonia .
  Fact 0.0: The military convention between Bulgaria and the Central Powers laid down the general plan for Bulgaria's campaign against the Kingdom of Serbia.
  Fact 0.1: The military convention severely limited the control of the Bulgarian High Command over the Bulgarian 1st Army.
  Fact 0.2: The Bulgarian 1st Army was designated part of a combined German, Bulgarian and Austro-Hungarian force.
  Fact 0.3: The combined force was commanded by Field Marshal August von Mackensen.
  Fact 0.4: Mackensen had recently led the German and Austro-Hungarian armies in the highly successful and victorious Gorlice-Tarnów Offensive of the Central Powers against the Russian army on the Eastern Front.
  Fact 0.5: Mackensen's army group was created specifically to wage war against the Serbian army in the pre-1913 borders of the country ('Old Serbia').
  Fact 0.6: Mackensen's army group was created to defeat the Serbian army wherever it found it.
  Fact 0.7: Mackensen's army group was created to open the land route between Hungary and Bulgaria.
  Fact 0.8: As commander, Mackensen acted independently.
  Fact 0.9: Mackensen received his directives only from the German High Command.
  Fact 0.10: The field marshal's orders to his Bulgarian forces had to be relayed to the commander of the 1st Army by the Bulgarian General Staff.
  Fact 0.11: The relay of orders through the Bulgarian General Staff left room for the Bulgarian General Staff to intervene when needed.
  Fact 0.12: According to the convention, the Bulgarian commander-in-chief retained full and direct control over the Bulgarian 2nd Army.
  Fact 0.13: According to the convention, the Bulgarian commander-in-chief retained full and direct control over the 2nd Army's operations in Vardar Macedonia.

Retrieved items:
<item 1>
Bulgaria during World War I
In November , while the decisive Serbian defeat unfolded , the French attempted to exert pressure on the Bulgarian 2nd Army , but were soon forced to halt their
attempts to drive north . The forces of General Maurice Sarrail that consisted of three French and one British division dug in along an 80 - kilometer front
from the river Cherna to Lake Doiran . With the fall of Pristina , Generail Sarrail realized that the Allies could no longer help the Serbians and decided to
begin pulling back his forces to Salonika . The Bulgarian High Command shifted its focus to the Allies in Macedonia and decided that the time was right to go on
the offensive . Several days were lost , however , in scouting , and it was only on 3 December that the 2nd army commenced a general advance . Nonetheless , the
French were able to retreat in good order towards Salonika . They were soon followed by the British , who were defeated at the Battle of Kosturino . On 11
December , the Bulgarian divisions reached the Greek border , where they were ordered to halt and warned repeatedly not to cross .
</item 1>

<item 2>
The Allies’ renewed offer failed to guarantee Serbian land concessions and omitted Southern Dobrudja, strengthening Bulgaria’s view that the Entente could not
control its Balkan allies. Bulgaria moved toward the Central Powers: Colonel Petar Ganchev went to Germany to negotiate a military convention, pro-German Nikola
Zhekov replaced War Minister Ivan Fichev, and Radoslavov sought Ottoman concessions. The Ottomans would negotiate only after Bulgaria aligned with the Central
Powers. Entente efforts to keep
</item 2>

<item 3>
After Italy joined the Entente in May 1915, Bulgaria’s strategic position and strong army made it a crucial prize for both sides. Entente diplomacy, led by
Britain, sought to win Bulgaria by offering Vardar Macedonia, with Russian support, while Central Powers also courted Sofia. Bulgaria could strengthen Serbia
and Russia and threaten the Ottomans for the Entente—or defeat Serbia, isolate Russia, and open the route to Constantinople for the Central Powers. Prime
Minister Radoslavov
</item 3>

<item 4>
In 1915, Allied leaders disagreed over how much territory to promise Bulgaria: Britain’s Grey urged restraint, while Lloyd George, Churchill, France, and Russia
favored concessions largely at Greece’s expense. Russia wanted a quick decision, but the Allies missed their best chance in spring. During summer, both sides
courted Bulgaria; the Allies funded opposition groups, while the Central Powers promised Bulgarian Balkan hegemony after Serbia’s defeat. Central Powers’
successes—including the Gallipoli stalemate and Russian retreat
</item 4>

<item 5>
After Austria-Hungary’s defeat at Cer, German and Austro-Hungarian efforts failed to move Bulgaria from its firm position. Russia sought a Balkan alliance with
Serbia, Montenegro, and Bulgaria, proposing Serbian territorial concessions for Bulgarian neutrality or cooperation. Serbia resisted, prioritizing control of
the Vardar and its strategic position over territorial concessions. Russia nevertheless offered Bulgaria vague gains; France, which preferred Romania, also
considered offering Bulgaria Eastern Thrace up to the Midia–Enos line
</item 5>

<item 6>
Bulgaria during World War I
Allied representatives met with the leaders of the Bulgarian opposition parties and also provided generous financial support for opposition newspapers ; they
even attempted to bribe high - ranking government officials . Germany and Austria - Hungary were not willing to remain on the sidelines and dispatched to
Bulgaria Duke John Albert of Mecklenburg , the former ambassador to the Ottoman Empire Hans Freiherr von Wangenheim and Prince Hohenlohe , who openly declared
that after the defeat of Serbia , Bulgaria would assume hegemony of the Balkans . What seized Bulgarian interest the most was indeed the balance of military
power . The situation on the major European fronts was at that time developing markedly in favor of the Central Powers , and while the Allied operation in
Gallipoli turned into a costly stalemate , the Russians were being driven out of Galicia and Poland . Under these circumstances . the Central Powers were hoping
to secure Bulgaria at last . Still , it took Entente diplomacy more than a month to give an answer to Radoslavov 's questions and the reply proved far from
satisfactory . In reality , it hardly differed from the offer the Allies presented in May .
</item 6>

<item 7>
Russia and France initially underestimated the Bulgarian-German negotiations, realizing in April that Entente pressure was pushing Bulgaria toward the Central
Powers. Russia’s ambassador urged opposition to Prime Minister Radoslavov and offered Tsar Ferdinand a French loan if Radoslavov were removed; Banque Perier
also proposed financing. These late efforts failed. In July 1914, German banks led by Disconto-Gesellschaft provided Bulgaria a 500-million-golden-leva loan, in
two installments
</item 7>

<item 8>
In the First Balkan War, Bulgaria helped drive the Ottomans back toward Constantinople, defeated their 1913 counteroffensive, and captured Adrianople. Romania
then demanded territory for its wartime neutrality. In World War I, Bulgaria joined the Central Powers and helped occupy Serbia by December 1915, achieving its
goal of conquering Vardar Macedonia. Its army committed about 424,375 men and suffered roughly 37,000 casualties. Serbia’s defeat strengthened the Central
Powers, securing
</item 8>

<item 9>
Bulgaria during World War I
By the middle of December , the entire Kingdom of Serbia was occupied by the armies of the Central Powers and the Allies were pushed back to Salonika by the
Bulgarians . Within two months of its entry into the war Bulgaria , achieved its main war goal : the conquest of Vardar Macedonia . For the duration of military
operations against Serbia and the Entente in 1915 , the Bulgarian Army committed a total of around 424,375 men , while its casualties were held to around 37,000
men . By the end of 1915 , the Central Powers had established firm and unbroken control over a vast territory that stretched from the North Sea to Mesopotamia .
They also drew great political and military dividends from the defeat and occupation of Serbia . Bulgaria conquered almost all the territory it desired ,
Germany gained unrestricted access to the natural resources of Ottoman Asia , the Ottomans received much needed German matériel assistance and Austria - Hungary
secured its southern flank and could completely focus its attention on the Russian and Italian fronts .
</item 9>

<item 10>
Bulgaria during World War I
Strategic geographic location and a strong military establishment made the country a desired ally for both warring coalitions , but Bulgaria 's regional
aspirations were difficult to satisfy because they included territorial claims against four Balkan countries . As the war progressed , the Central Powers of
Austria - Hungary and the German Empire found themselves in a better position to fulfil Bulgarian demands and persuaded the country to join their cause in
September 1915 . Though the smallest member of the Central Powers in area and in population , Bulgaria made vital contributions to their common war effort . Its
entry to the war heralded the defeat of Serbia , thwarted the foreign - policy goals of Romania , and ensured the continuation of the Ottoman war effort by
providing a geographical conduit for material assistance from Germany to Istanbul . Though the Balkan theatre of the war saw successful campaigns of rapid
movement in 1915 and 1916 , the conflict degraded into a state of attritional trench warfare on both the Northern and the Southern Bulgarian Fronts after most
Bulgarian territorial aspirations had been satisfied . This period of the war substantially weakened the Bulgarian economy , created various supply problems and
reduced the health and morale of Bulgarian troops on the front lines .
</item 10>
=== END ITEM id=7d025170afbe343f ===

=== ITEM id=099e13791a965cab ===
Gold sentences and facts:
Sentence 0: Lynne is the eldest of the Slater sisters and she arrives in Albert Square in 2000 with her family : father Charlie ( Derek Martin ) ; sisters Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) and Zoe ( Michelle Ryan ) , and grandmother Mo ( Laila Morse ) . Unlike Little Mo and Zoe , Lynne is aware that Zoe is in fact Kat 's daughter and helps keep the family secret until Kat reveals the truth to Zoe .
  Fact 0.0: Lynne is the eldest of the Slater sisters.
  Fact 0.1: Lynne arrives in Albert Square in 2000.
  Fact 0.2: Lynne arrives in Albert Square with her family.
  Fact 0.3: Lynne's father is Charlie, played by Derek Martin.
  Fact 0.4: Lynne's sister Kat is played by Jessie Wallace.
  Fact 0.5: Lynne's sister Little Mo is played by Kacey Ainsworth.
  Fact 0.6: Lynne's sister Zoe is played by Michelle Ryan.
  Fact 0.7: Lynne's grandmother is Mo, played by Laila Morse.
  Fact 0.8: Unlike Little Mo and Zoe, Lynne is aware that Zoe is in fact Kat's daughter.
  Fact 0.9: Lynne helps keep the family secret that Zoe is Kat's daughter until Kat reveals the truth to Zoe.

Retrieved items:
<item 1>
Lynne Hobbs
Lynne is a fierce judge of right and wrong , although she is not always able to practise what she preaches . She has a turbulent relationship with husband Garry
( Ricky Groves ) , but despite her complaints about him , it ends up being Lynne 's adultery that puts their relationship in jeopardy long before Garry strays .
Contents ( hide ) 1 Storylines 1.1 Backstory 1.2 2000 -- 2004 2 Creation and development 3 In popular culture 4 References 5 External links Storylines ( edit )
Backstory ( edit ) Lynne is the eldest of the Slater sisters and she arrives in Albert Square in 2000 with her family : father Charlie ( Derek Martin ) ;
sisters Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) and Zoe ( Michelle Ryan ) , and grandmother Mo ( Laila Morse ) . Unlike Little Mo and Zoe , Lynne
is aware that Zoe is in fact Kat 's daughter and helps keep the family secret until Kat reveals the truth to Zoe .
</item 1>

<item 2>
Lynne Hobbs
Syed Masood Tamwar Masood Zainab Masood Lilly Mattock Josie McFarlane Poppy Meadow Darren Miller Demi Miller Keith Miller Mickey Miller Rosie Miller Rob Minter
Archie Mitchell Ben Mitchell Danny Mitchell Glenda Mitchell Grant Mitchell Jamie Mitchell Kate Mitchell Little Mo Mitchell Peggy Mitchell Ronnie Mitchell Roxy
Mitchell Sam Mitchell Tiffany Mitchell Vinnie Monks Alfie Moon Anthony Moon Danny Moon Eddie Moon Jake Moon Michael Moon Nana Moon Spencer Moon Tyler Moon
Trevor Morgan Rod Norman Andy O'Brien Mercy Olubunmi Ali Osman Mehmet Osman Sue Osman Steve Owen Annie Palmer George Palmer Lola Pearce Minty Peterson Frankie
Pierre Paul Priestly Irene Raymond Louise Raymond Simon Raymond Terry Raymond Dennis Rickman Binnie Roberts Roly Willy Roper Matthew Rose Colin Russell Mandy
Salter Trevor Short Lydia Simmonds Ethel Skinner Charlie Slater Sean Slater Zoe Slater Leon Small Babe Smith Gus Smith Mary Smith Dan Sullivan Gavin Sullivan
Dawn Swann Celestine and Etta Tavernier Clyde Tavernier Hattie Tavernier Jules Tavernier Kelly Taylor Heather Trott Anthony Trueman Paul Trueman Yolande Trueman
Owen Turner Angie Watts Chrissie Watts Den Watts Wellard Carl White Carly Wicks David Wicks Dean Wicks Joe
</item 2>

<item 3>
Lynne Hobbs
links ( edit ) Lynne Hobbs at BBC Online EastEnders characters Present and future characters Kathy Beale Sharon Watts Ian Beale Dot Cotton Martin Fowler Phil
Mitchell Sonia Fowler Robbie Jackson Mel Owen Billy Mitchell Mo Harris Kat Slater Patrick Trueman Louise Mitchell Bobby Beale Stacey Slater Jean Slater Honey
Mitchell Denise Fox Max Branning Janet Mitchell Shirley Carter Jay Brown Rainie Cross Masood Ahmed Jack Branning Whitney Dean Tiffany Butcher Kim Fox Cora Cross
Tina Carter Linda Carter Mick Carter Charlie Cotton ( hide ) Past characters AJ Ahmed Della Alexander Johnny Allen Ruby Allen Aunt Sal Tom Banks Geoff Barnes
Clare Bates Debbie Bates Laurie Bates Nigel Bates Cindy Beale Jane Beale Laura Beale Lou Beale Lucy Beale Pete Beale Peter Beale Steven Beale Polly Becker Adam
Best Manda Best Abi Branning Alice Branning April Branning Bradley Branning Derek Branning Jim Branning Lauren Branning Suzy Branning Tanya Branning Aidan
Brosnan Diane Butcher Frank Butcher Janine Butcher Liam Butcher Mo Butcher Pat Butcher Ricky Butcher Kelvin Carpenter Tony Carpenter Nancy Carter Stan Carter
Zsa Zsa Carter Lorna Cartwright Barry Clark Christian Clarke Paul Coker Richard Cole Julie Cooper Ashley Cotton Charlie Cotton ( 1986
</item 3>

<item 4>
Lynne Hobbs
2000 -- 2004 ( edit ) Lynne is in a relationship with hapless mechanic Garry Hobbs ( Ricky Groves ) and spends much of her time despairing over Garry 's
inability to fully commit to their relationship . All Lynne really wants is to marry Garry , settle down and raise a family but while Garry loves Lynne , he is
reluctant to lose his `` bachelor status '' and he spends much of his time dodging Lynne 's ' hints ' about marriage . However , Lynne ( and the rest of the
Slaters ) wear Garry down and the two eventually get engaged . Lynne works in Ian Beale 's ( Adam Woodyatt ) café and often clashes with co-worker , Janine
Butcher ( Charlie Brooks ) , who regularly slacks off work and flirts with Garry to irritate her . This culminates in Lynne eventually getting Janine fired and
the two become enemies thereafter . In 2001 , Lynne becomes friends with womanising club owner Beppe di Marco ( Michael Greco ) and she often babysits his son ,
Joe ( Jake Kyprianou ) .
</item 4>

<item 5>
Lynne Hobbs
Barry Clark Christian Clarke Paul Coker Richard Cole Julie Cooper Ashley Cotton Charlie Cotton ( 1986 ) Dotty Cotton Nick Cotton Stella Crawford Beppe di Marco
Bruno and Luisa di Marco Gianni di Marco Rosa di Marco Ray Dixon Jase Dyer Huw Edwards Steve Elliot Nellie Ellis Barry Evans Natalie Evans Roy Evans Fatboy
Ferreira family Conor Flaherty Mary Flaherty Fred Fonseca Arthur Fowler Lisa Fowler Mark Fowler Michelle Fowler Pauline Fowler Ruth Fowler Vicki Fowler Chelsea
Fox Libby Fox Joanne Francis Jodie Gold Vanessa Gold Marge Green Nina Harris Dexter Hartman Julie Haye Alex Healy Christine Hewitt Ted Hills Tony Hills Garry
Hobbs Lynne Hobbs Lofty Holloway Vincent Hubbard Andy Hunter Alan Jackson Bianca Jackson Billie Jackson Blossom Jackson Carmel Jackson Carol Jackson Naima
Jeffery Saeed Jeffery Al Jenkins Greg Jessop Lucas Johnson Danielle Jones Naomi Julien Gita Kapoor Sanjay Kapoor Karim family Yusef Khan Tony King Harold Legg
Donna Ludlow Joe Macer Ryan Malloy Jill Marsden Afia Masood Amira Masood Shabnam Masood Syed Masood Tamwar Masood Zainab Masood Lilly Mattock Josie
</item 5>

<item 6>
Lynne Hobbs
Lordan was cast in the role of Lynne after the character was developed at an improvisation session for thirty actors and actresses earlier in the year . The
character was described as `` feisty '' , `` a loud mouth '' and `` lazy '' . On 15 December 2003 , it was announced that Lynne would be departing the following
year . The decision was a mutual decision between Lordan and show producers and Lynne departed in July 2004 . In popular culture ( edit ) The character of Lynne
Hobbs has been spoofed in the cartoon sketch show 2DTV . Also , impressionist Alistair McGowan has impersonated Lynne working in the cafe , in his comedy show ,
The Big Impression . References ( edit ) ^ Jump up to : `` Seven move in to Albert Square '' . BBC News Online . BBC Online . 30 August 2000 . Retrieved 7
September 2010 . Jump up ^ http://news.bbc.co.uk/1/hi/entertainment/tv_and_radio/3321345.stm External links ( edit ) Lynne Hobbs at BBC Online EastEnders
characters Present and future characters Kathy
</item 6>

<item 7>
Lynne Hobbs
Chrissie Watts Den Watts Wellard Carl White Carly Wicks David Wicks Dean Wicks Joe Wicks Kevin Wicks Lorraine Wicks Simon Wicks Debbie Wilkins Cindy Williams
James Willmott - Brown Willy May Wright Lists of characters By year : 1985 1986 2017 2018 Others : Flaherty family Dickens Hill inmates The Banned band members
Characters from spin - offs Retrieved from `` https://en.wikipedia.org/w/index.php?title=Lynne_Hobbs&oldid=838104750 '' Categories : EastEnders characters
Fictional characters introduced in 2000 Female characters in television Hidden categories : Articles needing additional references from March 2011 All articles
needing additional references All articles with unsourced statements Articles with unsourced statements from January 2011 Use dmy dates from March 2011 Talk
Contents About Wikipedia Gaeilge Edit links This page was last edited on 24 April 2018 , at 23 : 33 . About Wikipedia
</item 7>

<item 8>
Lynne Hobbs
Unsourced material may be challenged and removed . ( March 2011 ) ( Learn how and when to remove this template message ) Lynne Hobbs EastEnders character
Portrayed by Elaine Lordan Duration 2000 -- 2004 First appearance Episode 2003 18 September 2000 ( 2000 - 09 - 18 ) Last appearance Episode 2756 2 July 2004 (
2004 - 07 - 02 ) Created by Tony Jordan Introduced by John Yorke Spin - off appearances Slaters in Detention ( 2003 ) Classification Former ; regular Profile
Other names Lynne Slater Occupation Café worker ( show ) Family Father Charlie Slater Mother Viv Slater Sisters Kat Slater Little Mo Mitchell Belinda Peacock
Husband Garry Hobbs ( 2001 -- 2004 ) Daughters Vivienne Hobbs Grandmothers Mo Harris Uncles Harry Slater Nephews Freddie Mitchell Nieces Zoe Slater Lynne Hobbs
( also Slater ) is a fictional character from the BBC soap opera EastEnders , played by Elaine Lordan from September 2000 to July 2004 .
</item 8>

<item 9>
Lynne Hobbs
In January 2016 , Lynne attends Charlie 's funeral off - screen . Creation and development ( edit ) In 2000 , EastEnders ' executive producer John Yorke decided
to introduce the `` classic '' Slater family . He felt the show needed to go back to its roots and bring back some traditional values . BBC 's head of drama ,
Mal Young commented , `` We do not have enough solid families in the soap , there were a lot of fractured families and people who were alone . '' The family
were created as a replacement for the di Marco family , who were axed by Yorke . The family consisted of , grandmother Mo Harris ( Laila Morse ) , father
Charlie ( Derek Martin ) , his four children , Lynne ( Elaine Lordan ) , Kat ( Jessie Wallace ) , Little Mo ( Kacey Ainsworth ) , Zoe ( Michelle Ryan ) , as
well as Lynne 's boyfriend Garry Hobbs ( Ricky Groves ) . It later transpired that Kat was the mother of Zoe , after being raped by her uncle Harry Slater (
Michael Elphick ) .
</item 9>

<item 10>
Lynne Hobbs
In 2003 , another affair jeopardises Lynne 's marriage , but this time it is Garry who strays . Garry has drunken sex with Laura Beale ( Hannah Waterman ) and
she gets pregnant . Laura 's husband , Ian , reveals to her that he has had a vasectomy so the baby can not be his and throws her out so she turns to Garry for
support , thinking he is her baby 's father . Lynne has been trying for a baby and struggles to forgive Garry for having a baby with another woman , throwing
him out so he moves in with Laura to support her during her pregnancy . Lynne finally comes round and forgives Garry after she gets jealous when he has sex with
Lynne 's sister , Belinda Peacock ( Leanne Lakey ) . Lynne and Garry reconcile and in 2004 , Lynne becomes pregnant and a scan confirms that she is expecting a
girl , that they plan to name her Vivienne after Lynne 's late mother , Viv ( April Martin ) . However , disaster strikes when a fairground ride collapses in
Albert Square and Lynne is trapped under the debris .
</item 10>
=== END ITEM id=099e13791a965cab ===

=== ITEM id=0b4b9ce786cc7ec7 ===
Gold sentences and facts:
Sentence 0: Human fertilization is the union of a human egg and sperm , usually occurring in the ampulla of the fallopian tube . The result of this union is the production of a zygote cell , or fertilized egg , initiating prenatal development . Scientists discovered the dynamics of human fertilization in the nineteenth century .
  Fact 0.0: Human fertilization is the union of a human egg and sperm.
  Fact 0.1: Human fertilization usually occurs in the ampulla of the fallopian tube.
  Fact 0.2: The result of the union of a human egg and sperm is a zygote cell, also called a fertilized egg.
  Fact 0.3: The union of a human egg and sperm initiates prenatal development.
  Fact 0.4: Scientists discovered the dynamics of human fertilization in the nineteenth century.
Sentence 1: The process of fertilization involves a sperm fusing with an ovum . The most common sequence begins with ejaculation during copulation , follows with ovulation , and finishes with fertilization . Various exceptions to this sequence are possible , including artificial insemination , in vitro fertilization , external ejaculation without copulation , or copulation shortly after ovulation . Upon encountering the secondary oocyte , the acrosome of the sperm produces enzymes which allow it to burrow through the outer jelly coat of the egg . The sperm plasma then fuses with the egg 's plasma membrane , the sperm head disconnects from its flagellum and the egg travels down the Fallopian tube to reach the uterus .
  Fact 1.0: The process of fertilization involves a sperm fusing with an ovum.
  Fact 1.1: The most common sequence of fertilization begins with ejaculation during copulation.
  Fact 1.2: The most common sequence of fertilization is followed by ovulation.
  Fact 1.3: The most common sequence of fertilization finishes with fertilization.
  Fact 1.4: Various exceptions to the most common sequence of fertilization are possible.
  Fact 1.5: One exception to the sequence of fertilization is artificial insemination.
  Fact 1.6: One exception to the sequence of fertilization is in vitro fertilization.
  Fact 1.7: One exception to the sequence of fertilization is external ejaculation without copulation.
  Fact 1.8: One exception to the sequence of fertilization is copulation shortly after ovulation.
  Fact 1.9: Upon encountering the secondary oocyte, the acrosome of the sperm produces enzymes.
  Fact 1.10: The enzymes from the acrosome of the sperm allow the sperm to burrow through the outer jelly coat of the egg.
  Fact 1.11: The sperm plasma fuses with the egg's plasma membrane.
  Fact 1.12: The sperm head disconnects from its flagellum.
  Fact 1.13: The egg travels down the Fallopian tube to reach the uterus.

Retrieved items:
<item 1>
Gamete
When mature the haploid ovule produces the female gamete which are ready for fertilization . The male haploid is pollen and is produced by the anther , when
pollen lands on a mature stigma of a flower it grows a pollen tube down into the flower . The haploid pollen then produces sperm by mitosis and releases them
for fertilization . Notes and references ( edit ) Jump up ^ `` gamete '' . Online Etymology Dictionary . Jump up ^ Marshall , A.M. 1893 . Vertebrate embryology
: a text - book for students and practitioners . GP Putnam 's Sons . Jump up ^ Yeung , C. , M. Anapolski , M. Depenbusch , M. Zitzmann , and T. Cooper . 2003 .
`` Human sperm volume regulation . Response to physiological changes in osmolality , channel blockers and potential sperm osmolytes . '' Human Reproduction 18 :
1029 . ^ Jump up to : Jay Phelan ( 30 April 2009 ) . What Is Life ? : A Guide to Biology W / Prep - U . Macmillan . pp.
</item 1>

<item 2>
Human fertilization
This is why all mitochondria in humans are of maternal origin . Still , a considerable amount of RNA from the sperm is delivered to the resulting embryo and
likely influences embryo development and the phenotype of the offspring . Replication ( edit ) The pronuclei migrate toward the center of the oocyte , rapidly
replicating their DNA as they do so to prepare the zygote for its first mitotic division . Mitosis ( edit ) Usually 23 chromosomes from spermatozoon and 23
chromosomes from egg cell fuse ( half of spermatozoons carry X chromosome and the other half Y chromosome ) . Their membranes dissolve , leaving no barriers
between the male and female chromosomes . During this dissolution , a mitotic spindle forms between them . The spindle captures the chromosomes before they
disperse in the egg cytoplasm . Upon subsequently undergoing mitosis ( which includes pulling of chromatids towards centrioles in anaphase ) the cell gathers
genetic material from the male and female together . Thus , the first mitosis of the union of sperm and oocyte is the actual fusion of their chromosomes .
</item 2>

<item 3>
Human fertilization
Human fertilization - wikipedia Human fertilization Jump to : navigation , search Part of a series on Human growth and development Stages Human embryogenesis
Fetus Infant Toddler Early childhood Child Preadolescence Adolescence Adult Middle age Old age Biological milestones Fertilization Childbirth Walking Language
acquisition Puberty Menopause Ageing Death Development and psychology Pre - and perinatal Infant and child Adolescent Youth Young adult Adult Maturity
Developmental stage theories Attachment Ecological Psychosocial Psychosexual development Moral Cognitive Cultural - historical Evolutionary Human body portal
The acrosome reaction for a sea urchin , a similar process . Note that the picture shows several stages of one and the same spermatozoon - only one penetrates
the ovum Illustration depicting ovulation and fertilization . Human fertilization is the union of a human egg and sperm , usually occurring in the ampulla of
the fallopian tube . The result of this union is the production of a zygote cell , or fertilized egg , initiating prenatal development . Scientists discovered
the dynamics of human fertilization in the nineteenth century . The process of fertilization involves a sperm fusing with an ovum .
</item 3>

<item 4>
Human fertilization
The process of fertilization involves a sperm fusing with an ovum . The most common sequence begins with ejaculation during copulation , follows with ovulation
, and finishes with fertilization . Various exceptions to this sequence are possible , including artificial insemination , in vitro fertilization , external
ejaculation without copulation , or copulation shortly after ovulation . Upon encountering the secondary oocyte , the acrosome of the sperm produces enzymes
which allow it to burrow through the outer jelly coat of the egg . The sperm plasma then fuses with the egg 's plasma membrane , the sperm head disconnects from
its flagellum and the egg travels down the Fallopian tube to reach the uterus . In vitro fertilization ( IVF ) is a process by which egg cells are fertilized by
sperm outside the womb , in vitro . Contents ( hide ) 1 Anatomy 1.1 Corona radiata 1.2 Cone of attraction and perivitelline membrane 1.3 Sperm preparation 1.4
Zona pellucida 1.4.
</item 4>

<item 5>
Human fertilization
Jump up ^ `` Fertilization '' . Retrieved 28 July 2010 . Jump up ^ Angier , Natalie ( 2007 - 06 - 12 ) . `` Sleek , Fast and Focused : The Cells That Make Dad
Dad '' . The New York Times . Jump up ^ Suzanne Wymelenberg , Science and Babies , National Academy Press , page 17 Jump up ^ Richard E. Jones and Kristin H.
Lopez , Human Reproductive Biology , Third Edition , Elsevier , 2006 , page 238 Jump up ^ `` Fertilization : The Cortical Reaction '' . Boundless . Boundless .
Retrieved 14 March 2013 . Jump up ^ Jodar , M. ; Selvaraju , S. ; Sendler , E. ; Diamond , M.P. ; Krawetz , S.A. ; for the Reproductive Medicine Networks ( 2013
) . `` The presence , role and clinical use of spermatozoal RNAs '' . Human Reproduction Update. 19 ( 6 ) : 604 -- 624 .
</item 5>

<item 6>
Human fertilization
Sperm preparation ( edit ) Further information : Acrosome reaction At the beginning of the process , the sperm undergoes a series of changes , as freshly
ejaculated sperm is unable or poorly able to fertilize . The sperm must undergo capacitation in the female 's reproductive tract over several hours , which
increases its motility and destabilizes its membrane , preparing it for the acrosome reaction , the enzymatic penetration of the egg 's tough membrane , the
zona pellucida , which surrounds the oocyte . Zona pellucida ( edit ) After binding to the corona radiata the sperm reaches the zona pellucida , which is an
extra-cellular matrix of glycoproteins . A special complementary molecule on the surface of the sperm head binds to a ZP3 glycoprotein in the zona pellucida .
This binding triggers the acrosome to burst , releasing enzymes that help the sperm get through the zona pellucida . Some sperm cells consume their acrosome
prematurely on the surface of the egg cell , facilitating the penetration by other sperm cells .
</item 6>

<item 7>
Human fertilization
Fusion ( edit ) Fertilization and implantation in humans . After the sperm enters the cytoplasm of the oocyte ( also called ovocyte ) , the tail and the outer
coating of the sperm disintegrate and the cortical reaction takes place , preventing other sperm from fertilizing the same egg . The oocyte now undergoes its
second meiotic division producing the haploid ovum and releasing a polar body . The sperm nucleus then fuses with the ovum , enabling fusion of their genetic
material . Cell membranes ( edit ) The fusion of cell membranes of the secondary oocyte and sperm takes place . Transformations ( edit ) In preparation for the
fusion of their genetic material both the oocyte and the sperm undergo transformations as a reaction to the fusion of cell membranes . The oocyte completes its
second meiotic division . This results in a mature ovum . The nucleus of the oocyte is called a pronucleus in this process , to distinguish it from the nuclei
that are the result of fertilization . The sperm 's tail and mitochondria degenerate with the formation of the male pronucleus . This is why all mitochondria in
humans are of maternal origin .
</item 7>

<item 8>
Human fertilization
1 Cortical reaction 2 Fusion 2.1 Cell membranes 2.2 Transformations 2.3 Replication 2.4 Mitosis 3 Fertilization age 4 Diseases 5 See also 6 References 7
External links Anatomy ( edit ) Corona radiata ( edit ) The sperm binds through the corona radiata , a layer of follicle cells on the outside of the secondary
oocyte . Fertilization occurs when the nucleus of both a sperm and an egg fuse to form a diploid cell , known as zygote . The successful fusion of gametes forms
a new organism . Cone of attraction and perivitelline membrane ( edit ) Where the spermatozoon is about to pierce , the yolk ( ooplasm ) is drawn out into a
conical elevation , termed the cone of attraction or reception cone . Once the spermatozoon has entered , the peripheral portion of the yolk changes into a
membrane , the perivitelline membrane , which prevents the passage of additional spermatozoa .
</item 8>

<item 9>
Human fertilization
As a population , sperm cells have on average 50 % genome similarity so the premature acrosomal reactions aid fertilization by a member of the same cohort . It
may be regarded as a mechanism of kin selection . Recent studies have shown that the egg is not passive during this process . Cortical reaction ( edit ) Once
the sperm cells find their way past the zona pellucida , the cortical reaction occurs . Cortical granules inside the secondary oocyte fuse with the plasma
membrane of the cell , causing enzymes inside these granules to be expelled by exocytosis to the zona pellucida . This in turn causes the glyco - proteins in
the zona pellucida to cross-link with each other -- i.e. the enzymes cause the ZP2 to hydrolyse into ZP2f -- making the whole matrix hard and impermeable to
sperm . This prevents fertilization of an egg by more than one sperm . The cortical reaction and acrosome reaction are both essential to ensure that only one
sperm will fertilize an egg . Fusion ( edit ) Fertilization and implantation in humans .
</item 9>

<item 10>
Human fertilization
Each of the two daughter cells resulting from that mitosis has one replica of each chromatid that was replicated in the previous stage . Thus , they are
genetically identical . Fertilization age ( edit ) Fertilization is the event most commonly used to mark the zero point in descriptions of prenatal development
of the embryo or fetus . The resultant age is known as fertilization age , fertilizational age , embryonic age , fetal age or ( intrauterine ) developmental (
IUD ) age . Gestational age , in contrast , takes the beginning of the last menstrual period ( LMP ) as the zero point . By convention , gestational age is
calculated by adding 14 days to fertilization age and vice versa . In fact , however , fertilization usually occurs within a day of ovulation , which , in turn
, occurs on average 14.6 days after the beginning of the preceding menstruation ( LMP ) . There is also considerable variability in this interval , with a 95 %
prediction interval of the ovulation of 9 to 20 days after menstruation even for an average woman who has a mean LMP - to - ovulation time of 14.6 .
</item 10>
=== END ITEM id=0b4b9ce786cc7ec7 ===
