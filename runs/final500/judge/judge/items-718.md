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

=== ITEM id=a0f67f38dbe2803d ===
Gold sentences and facts:
Sentence 0: The current phrasing appears to have originated in the 1910 book Physics by Charles Riborg Mann and George Ransom Twiss . The question `` When a tree falls in a lonely forest , and no animal is near by to hear it , does it make a sound ? Why ? '' is posed along with many other questions to quiz readers on the contents of the chapter , and as such , is posed from a purely physical point of view .
  Fact 0.0: The current phrasing of the tree-falling question appears to have originated in the 1910 book Physics.
  Fact 0.1: The 1910 book Physics was written by Charles Riborg Mann and George Ransom Twiss.
  Fact 0.2: The question "When a tree falls in a lonely forest, and no animal is near by to hear it, does it make a sound? Why?" is posed along with many other questions to quiz readers on the contents of the chapter.
  Fact 0.3: The tree-falling question is posed from a purely physical point of view.
Sentence 1: Philosopher George Berkeley , in his work , A Treatise Concerning the Principles of Human Knowledge ( 1710 ) , proposes , `` But , say you , surely there is nothing easier than for me to imagine trees , for instance , in a park ( ... ) and nobody by to perceive them . ( ... ) The objects of sense exist only when they are perceived ; the trees therefore are in the garden ( ... ) no longer than while there is somebody by to perceive them . '' ( It is worth noting that the quote from section 45 is arguably a statement of an objection to Berkeley 's view , and not a proclamation of it . ) Nevertheless , Berkeley never actually wrote about the question .
  Fact 1.0: Philosopher George Berkeley wrote A Treatise Concerning the Principles of Human Knowledge, published in 1710.
  Fact 1.1: In A Treatise Concerning the Principles of Human Knowledge, Berkeley proposes a passage that begins "But, say you, surely there is nothing easier than for me to imagine trees, for instance, in a park ... and nobody by to perceive them."
  Fact 1.2: The passage quoted from Berkeley's treatise continues "The objects of sense exist only when they are perceived; the trees therefore are in the garden ... no longer than while there is somebody by to perceive them."
  Fact 1.3: The quote from section 45 of Berkeley's treatise is arguably a statement of an objection to Berkeley's view, not a proclamation of it.
  Fact 1.4: Berkeley never actually wrote about the tree-falling question.

Retrieved items:
<item 1>
If a tree falls in a forest
155 -- 158 . Jump up ^ Watts 1962 , pp. 111 -- 113 Jump up ^ A Buddhist Bible , `` Sutra Spoken by The Sixth Patriarch '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=If_a_tree_falls_in_a_forest&oldid=812540716 '' Categories : Fictional trees Riddles Philosophical problems Thought
experiments in philosophy of mind Talk Contents About Wikipedia Español فارسی עברית Македонски Nederlands Português Русский Svenska ไทย 中文 Edit links This page
was last edited on 28 November 2017 , at 12 : 14 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 1>

<item 2>
If a tree falls in a forest
If there be no ears to hear , there will be no sound . '' Albert Einstein is reported to have asked his fellow physicist and friend Niels Bohr , one of the
founding fathers of quantum mechanics , whether he realistically believed that ' the moon does not exist if nobody is looking at it . ' To this Bohr replied
that however hard he ( Einstein ) may try , he would not be able to prove that it does , thus giving the entire riddle the status of a kind of an infallible
conjecture -- one that can not be either proved or disproved . The current phrasing appears to have originated in the 1910 book Physics by Charles Riborg Mann
and George Ransom Twiss . The question `` When a tree falls in a lonely forest , and no animal is near by to hear it , does it make a sound ? Why ? '' is posed
along with many other questions to quiz readers on the contents of the chapter , and as such , is posed from a purely physical point of view . Metaphysics (
edit ) The possibility of unperceived existence ( edit ) Main article : George Berkeley Can something exist without being perceived ? -- e.g.
</item 2>

<item 3>
If a tree falls in a forest
`` The mind is like akasa ( space ) . Just as there are the objects in the akasa , so there are thoughts in the mind . The akasa is the counterpart of the mind
and objects are of thought . One can not hope to measure the universe and study the phenomena . It is impossible . For the objects are mental creations ... The
universe is only an object created by the mind and has its being in the mind . '' ( Talks with Sri Ramana Maharshi , pp. 478 -- 479 ) . `` The shrutis ( a
subdivision of the Vedas ) and the sages say that the objects are only mental creations . They have no substantive being . Investigate the matter and ascertain
the truth of the statement . The result will be the conclusion that the objective world is in the subjective consciousness . '' ( TWSRM , p. 479 ) . In popular
culture ( edit ) Canadian songwriter , social activist and environmentalist Bruce Cockburn poses the question in the chorus of his 1989 song `` If a Tree Falls
'' and frames it with his lyrics as a pressing question regarding the cause and effect of deforestation .
</item 3>

<item 4>
If a tree falls in a forest
So then , how is it known that ' sound as it is mechanically understood ' will occur if that sound is not perceived ? Buddhist perspectives ( edit ) Tibetan (
edit ) from The Thirty - Seven Bodhisattva Practices by Ngulchu Thogme Zangpo , `` 22 Appearances are one 's own mind . From the beginning , mind 's nature is
free from the extremes of elaboration . Knowing this , not to engage the mind in subject - object duality is the bodhisattva 's practice . '' Zen ( edit ) Hui -
neng 's flag ( edit ) There is a well - known story of Hui - neng , a well - respected Buddhist monk who later became known as the founder of the Zen school ,
who one day happened to be passing by two monks . `` Two monks were arguing about the temple flag waving in the wind . One said , `` The flag moves . '' The
other said , `` The wind moves . '' They argued back and forth but could not agree . The Sixth Ancestor said , `` Gentlemen !
</item 4>

<item 5>
If a tree falls in a forest
It may then be recognized that the unobserved event was absolutely identical to an event which did not occur at all . '' ( this apparent quote has no
attribution or reference and none can be found online with reasonable effort ) . Of course , the fact that the tree is known to have changed state from '
upright ' to ' fallen ' implies that the event must be observed to ask the question at all -- even if only by the supposed deaf onlooker . The British
philosopher of science Roy Bhaskar , credited with developing critical realism has argued , in apparent reference to this riddle , that : If men ceased to exist
sound would continue to travel and heavy bodies to fall to the earth in exactly the same way , though ex hypothesi there would be no - one to know it This
existence of an unobserved real is integral to Bhaskar 's ontology , which contends ( in opposition to the various strains of positivism which have dominated
both natural and social science in the twentieth century ) that ' real structures exist independently of and are often out of phase with the actual patterns of
events ' .
</item 5>

<item 6>
If a tree falls in a forest
Washington - state - based wilderness conservatory Northwest Trek used a shortened form of the quote in its mid-1970s television advertisement , as such : ``
There is no sound unless someone is there to see it or hear it . Experience it at Northwest Trek . '' See also ( edit ) Counterfactual definiteness Descartes '
dream argument Epistemology Object ( philosophy ) Observer effect ( physics ) Kōan Ontology Schrödinger 's cat Principle of locality References ( edit ) Jump up
^ A Treatise Concerning the Principles of Human Knowledge , 1734 . section 23 . Jump up ^ A Treatise Concerning the Principles of Human Knowledge , 1734 .
section 45 . Jump up ^ Whether Berkeley thought objects continued to exist when unperceived by finite minds , and if so , in what manner , is the subject of
serious debate among Berkeley scholars . Jump up ^ The Chautauquan , June 1883 , Volume 3 , Issue 9 , p. 543 Jump up ^ Scientific American , April 5 , 1884 , pg
218 .
</item 6>

<item 7>
If a tree falls in a forest
2.1 Hui - neng 's Flag 2.4. 2.2 Hui - neng Receives The Dharma 2.5 Hinduism 3 In popular culture 4 See also 5 References History ( edit ) Philosopher George
Berkeley , in his work , A Treatise Concerning the Principles of Human Knowledge ( 1710 ) , proposes , `` But , say you , surely there is nothing easier than
for me to imagine trees , for instance , in a park ( ... ) and nobody by to perceive them . ( ... ) The objects of sense exist only when they are perceived ;
the trees therefore are in the garden ( ... ) no longer than while there is somebody by to perceive them . '' ( It is worth noting that the quote from section
45 is arguably a statement of an objection to Berkeley 's view , and not a proclamation of it . ) Nevertheless , Berkeley never actually wrote about the
question . Some years later , a similar question is posed . It is unknown whether the source of this question is Berkeley or not .
</item 7>

<item 8>
If a tree falls in a forest
It is unknown whether the source of this question is Berkeley or not . In June 1883 in the magazine The Chautauquan , the question was put , `` If a tree were
to fall on an island where there were no human beings would there be any sound ? '' They then went on to answer the query with , `` No . Sound is the sensation
excited in the ear when the air or other medium is set in motion . '' This seems to imply that the question is posed not from a philosophical viewpoint , but
from a purely scientific one . The magazine Scientific American corroborated the technical aspect of this question , while leaving out the philosophic side , a
year later when they asked the question slightly reworded , `` If a tree were to fall on an uninhabited island , would there be any sound ? '' And gave a more
technical answer , `` Sound is vibration , transmitted to our senses through the mechanism of the ear , and recognized as sound only at our nerve centers . The
falling of the tree or any other disturbance will produce vibration of the air . If there be no ears to hear , there will be no sound . ''
</item 8>

<item 9>
If a tree falls in a forest
-- e.g. `` is sound only sound if a person hears it ? '' The most immediate philosophical topic that the riddle introduces involves the existence of the tree (
and the sound it produces ) outside of human perception . If no one is around to see , hear , touch or smell the tree , how could it be said to exist ? What is
it to say that it exists when such an existence is unknown ? Of course , from a scientific viewpoint , it exists . It is human beings that are able to perceive
it . George Berkeley in the 18th century developed subjective idealism , a metaphysical theory to respond to these questions , coined famously as `` to be is to
be perceived '' . Today meta - physicists are split . According to substance theory , a substance is distinct from its properties , while according to bundle
theory , an object is merely its sense data . The definition of sound , simplified , is a hearable noise . The tree will make a sound , even if nobody heard it
. The definition states that sound is a hearable noise . So the tree could have been heard , though nobody was around to do so .
</item 9>

<item 10>
If a tree falls in a forest
If a tree falls in a forest - wikipedia If a tree falls in a forest Jump to : navigation , search For the 2011 documentary , see If a Tree Falls : A Story of
the Earth Liberation Front . A fallen tree in a forest `` If a tree falls in a forest and no one is around to hear it , does it make a sound ? '' is a
philosophical thought experiment that raises questions regarding observation and perception . Contents ( hide ) 1 History 2 Metaphysics 2.1 The possibility of
unperceived existence 2.2 Knowledge of the unobserved world 2.3 The dissimilarity between sensation and reality 2.4 Buddhist perspectives 2.4. 1 Tibetan 2.4. 2
Zen 2.4. 2.1 Hui - neng 's Flag 2.4.
</item 10>
=== END ITEM id=a0f67f38dbe2803d ===

=== ITEM id=79363879a07d2129 ===
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
Fertilizer runoffs have resulted in eutrophication , with large algal blooms and elevated bacterial levels . By 2014 , large swaths of lake bed were exposed and
salt levels drastically increased due to mandated water transfers to metropolitan areas along the coast and other factors , limiting the water inflow . Besides
the resulting fish kills , the shrinking lake interrupts the bird migration , causes dust clouds , and impacts local tourism negatively . Remediation efforts (
edit ) Past efforts and proposals for a Sea level canal ( edit ) Map of New River and Alamo River courses Alternatives for `` saving '' the Salton Sea have been
evaluated since 1955 . The New River passes from Mexicali , Baja California , to the Imperial Valley , and on to the Salton Sea . Much of the current interest
in the sea was sparked in the 1990s by Congressman Sonny Bono . His widow , Mary Bono Mack , elected to fill his seat , has continued to be interested in the
Salton Sea , as has Representative Jerry Lewis of Redlands . In 1998 , the Sonny Bono Salton Sea Restoration Project was named for the politician .
</item 1>

<item 2>
Salton Sea
Ecology ( edit ) See also : Sonny Bono Salton Sea National Wildlife Refuge § Wildlife Fish population ( edit ) Due to the high salinity , very few fish species
can tolerate living in the Salton Sea . Introduced tilapia are the main fish that can tolerate the high salinity levels and pollution . Other freshwater fish
species live in the rivers and canals that feed the Salton Sea , including threadfin shad , carp , red shiner , channel catfish , white catfish , largemouth
bass , mosquitofish , sailfin molly , and the vulnerable desert pupfish . Avian population ( edit ) The Salton Sea has been termed a `` crown jewel of avian
biodiversity '' by Dr. Milt Friend of the Salton Sea Science Office . Over 400 species have been documented at the Salton Sea . The most diverse and probably
the most significant populations of bird life in the continental United States are hosted , rivaled only by Big Bend National Park in Texas . It supports 30 %
of the remaining population of the American white pelican . The Salton Sea is also a major resting stop on the Pacific Flyway .
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
1 Past efforts and proposals for a sea level canal 3.1. 2 State restoration plan 4 Earthquake geology 5 Water temperature 6 In popular culture 7 See also 8
References 9 Further reading 10 External links History ( edit ) The area was once part of a vast inland sea that covered a large area of Southern California .
Geologists estimate that for three million years , at least through all the years of the Pleistocene glacial age , a large delta was deposited by the Colorado
River in the southern region of the Imperial Valley . Eventually , the delta reached the western shore of the Gulf of California , creating a barrier that
separated the area of the Salton Sea from the northern reaches of the Gulf . Were it not for this barrier the entire Salton Sink along with the Imperial Valley
would be submerged as the Gulf would extend as far north as Indio . Since the exclusion of the ocean , the Salton Basin has over the ages been alternately a
freshwater lake , an increasingly saline endorheic lake , and a dry desert basin , depending on river flows and the balance between inflow and evaporative loss
.
</item 4>

<item 5>
Salton Sea
Surface water temperature ( Sandy Beach ) Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec 57 ° F 14 ° C 57 ° F 14 ° C 64 ° F 18 ° C 72 ° F 22 ° C 77 ° F 25 ° C
82 ° F 28 ° C 90 ° F 32 ° C 91 ° F 33 ° C 86 ° F 30 ° C 81 ° F 27 ° C 72 ° F 22 ° C 61 ° F 17 ° C In popular culture ( edit ) An abandoned boat stuck in the
ground , close to the west coast marina of the Salton Sea Films The documentary Plagues & Pleasures on the Salton Sea ( which premiered at the 2004 Slamdance
Film Festival and , with the addition of John Waters as narrator , premiered at the Provincetown International Film Festival in Massachusetts and opened in
select theaters on February 24 , 2006 ) , covers the first 100 years of the Salton Sea , along with the environmental issues and offbeat residents of the region
.
</item 5>

<item 6>
Salton Sea
The resulting outflow overwhelmed the engineered canal , and the river flowed into the Salton Basin for two years , filling the historic dry lake bed and
creating the modern sea , before repairs were completed . While it varies in dimensions and area with fluctuations in agricultural runoff and rainfall , the
Salton Sea is about 15 miles ( 24 km ) by 35 miles ( 56 km ) . With an estimated surface area of 343 square miles ( 890 km ) or 350 square miles ( 910 km ) ,
the Salton Sea is the largest lake in California . The average annual inflow is less than 1,200,000 acre feet ( 1.5 km ) , which is enough to maintain a maximum
depth of 43 feet ( 13 m ) and a total volume of about 6,000,000 acre feet ( 7.4 km ) . However , due to changes in water apportionments agreed upon for the
Colorado River under the Quantification Settlement Agreement of 2003 , the overall water level of the sea is expected to decrease significantly between 2013 and
2021 .
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
depth 43 ft ( 13 m ) Water volume 6,000,000 acre ⋅ ft ( 7.4 km ) Surface elevation − 236 ft ( − 71.9 m ) ( below sea level ) Settlements Bombay Beach , Desert
Beach , Desert Shores , Salton City , Salton Sea Beach , North Shore References U.S. Geological Survey Geographic Names Information System : Salton Sea Map of
the Salton Sea drainage area The Salton Sea is a shallow , saline , endorheic rift lake located directly on the San Andreas Fault , predominantly in California
's Imperial and Coachella valleys . The lake occupies the lowest elevations of the Salton Sink in the Colorado Desert of Imperial and Riverside Counties in
Southern California . Its surface is 236.0 ft ( 71.9 m ) below sea level as of January 2018 . The deepest point of the sea is 5 ft ( 1.5 m ) higher than the
lowest point of Death Valley . The sea is fed by the New , Whitewater , and Alamo rivers , as well as agricultural runoff , drainage systems , and creeks .
</item 8>

<item 9>
Salton Sea
Many of the species of fish that lived in the sea have been killed off by the combination of pollutants , salt levels , and algal blooms . Dead fish have been
known to wash up in mass quantities on the beaches . The smell of the lake , combined with the stench of the decaying fish , also contributed to the decline of
the tourist industry around the Salton Sea . Many people now visit the Salton Sea and the surrounding settlements to explore the abandoned structures . The town
of Niland is 1.5 miles ( 3 km ) southeast of the sea , with a population of 1,006 . Evidence of geothermal activity is also visible . Mudpots and mud volcanoes
are found on the eastern side of the Salton Sea . A number of geothermal electricity generation plants are located along the southeastern shore of the Salton
Sea in Imperial County . The US Geological Survey describes the smell as `` objectionable '' , `` noxious '' , `` unique '' , and `` pervasive '' .
</item 9>

<item 10>
Salton Sea
San Diego County Water Authority . Retrieved 2014 - 10 - 14 . Jump up ^ See http://www.usbr.gov/lc/region/programs/SaltonSea_2004-2014.xlsx , linked at
https://www2.usgs.gov/saltonsea/monitoring.html Jump up ^ Khaled M. Bali ( 5 October 2013 ) . `` Salton Sea and Salinity '' . UC Davis , Cooperative Extension
Imperial County . Retrieved 2015 - 06 - 29 . The salinity of the Pacific Ocean is approximately 35,000 mg / l ( 1 ppm = 1 mg / l ) . Colorado River water
salinity is about 650 -- 700 mg / L. Salton Sea salinity is about 44,000 mg / L , that is approximately 4.4 % salt . The amount of salts that is deposited in
the Imperial Valley agricultural land with irrigation water is approximately four million tons of salts annually . Jump up ^ Alles , DL ( 2007 - 08 - 06 ) . ``
Geology of the Salton Trough '' ( PDF ) .
</item 10>
=== END ITEM id=79363879a07d2129 ===

=== ITEM id=e5b79419ad4f22cf ===
Gold sentences and facts:
Sentence 0: India and Pakistan acquired independence in 1947 , but Governors - General continued to be appointed over each nation until republican constitutions were written . Louis Mountbatten , 1st Earl Mountbatten of Burma remained Governor - General of India for some time after independence , but the two nations were otherwise headed by native Governors - General . India became a secular republic in 1950 ; Pakistan became an Islamic one in 1956 .
  Fact 0.0: India and Pakistan gained independence in 1947.
  Fact 0.1: Governors-General continued to be appointed over each nation after 1947 until republican constitutions were written.
  Fact 0.2: Louis Mountbatten, 1st Earl Mountbatten of Burma, remained Governor-General of India for some time after independence.
  Fact 0.3: The two nations were otherwise headed by native Governors-General.
  Fact 0.4: India became a secular republic in 1950.
  Fact 0.5: Pakistan became an Islamic republic in 1956.

Retrieved items:
<item 1>
Governor-General of India
The power to elect the Governor - General was retained by the Court of Directors , but the choice became subject to the Sovereign 's approval . After the Indian
Rebellion of 1857 , the East India Company 's territories in India were put under the direct control of the Sovereign . The Government of India Act 1858 vested
the power to appoint the Governor - General in the Sovereign . The Governor - General , in turn , had the power to appoint all lieutenant governors in India ,
subject to the Sovereign 's approval . India and Pakistan acquired independence in 1947 , but Governors - General continued to be appointed over each nation
until republican constitutions were written . Louis Mountbatten , 1st Earl Mountbatten of Burma remained Governor - General of India for some time after
independence , but the two nations were otherwise headed by native Governors - General . India became a secular republic in 1950 ; Pakistan became an Islamic
one in 1956 . Functions ( edit ) Lord Curzon in his robes as Viceroy of India , a post he held from 1899 to 1905 .
</item 1>

<item 2>
Governor-General of India
.. Governors - General of India ( 1833 -- 1858 ) Bentinck Metcalfe Auckland Ellenborough Bird Hardinge Dalhousie Canning Governors - General and Viceroys of
India ( 1858 -- 1947 ) Canning Elgin Napier Denison Lawrence Mayo Strachey Napier Northbrook Lytton Ripon Dufferin Lansdowne Elgin Curzon Minto Penshurst
Chelmsford Reading Irwin Willingdon Wavell Mountbatten Governors - General of the Union of India ( 1947 -- 1950 ) Mountbatten Rajagopalachari Representatives of
the monarch in Commonwealth realms and Dominions National Counsellors of State United Kingdom Governors - general Antigua and Barbuda Australia Bahamas Barbados
Belize Canada Grenada Jamaica New Zealand Papua New Guinea Saint Kitts and Nevis Saint Lucia Saint Vincent and the Grenadines Solomon Islands Tuvalu
Sub-national Australia Governors New South Wales Queensland South Australia Tasmania Victoria Western Australia Administrator Northern Territory Canada
Lieutenant governors Alberta British Columbia Manitoba New Brunswick Newfoundland Nova Scotia Ontario Prince Edward Island Quebec Saskatchewan Realm of New
Zealand Governor - general Niue ,
</item 2>

<item 3>
Governor-General of India
The Chamber of Princes was an institution established in 1920 by a Royal Proclamation of King - Emperor George V to provide a forum in which the princely rulers
could voice their needs and aspirations to the government . The chamber usually met only once a year , with the Viceroy presiding , but it appointed a Standing
Committee , which met more often . Upon independence in August 1947 , the title of Viceroy was abolished . The representative of the British Sovereign became
known once again as the Governor - General . C. Rajagopalachari became the only Indian Governor - General . However , once India acquired independence , the
Governor - General 's role became almost entirely ceremonial , with power being exercised on a day - to - day basis by the Indian cabinet . After the nation
became a republic in 1950 , the President of India continued to perform the same functions .
</item 3>

<item 4>
Governor-General of India
Throughout the British administration , Governors - General retreated to the Viceregal Lodge ( Rashtrapati Niwas ) at Shimla each summer to escape the heat ,
and the government of India moved with them . The Viceregal Lodge now houses the Indian Institute of Advanced Study . Insignia ( edit ) See also : Star of India
( flag ) Badge of the Governor - General ( 1885 -- 1947 ) Standard of the Governor - General ( 1885 -- 1947 ) Standard of the Governor - General ( 1947 -- 50 )
See also ( edit ) India portal Pakistan portal United Kingdom portal Politics portal List of governors - general of India Commander - in - Chief , India British
Empire Emperor of India Indian independence movement Council of India British Raj Secretary of State for India India Office Indian Civil Service Partition of
India History of Bangladesh History of India History of Pakistan References ( edit ) Jump up ^ The term British India is mistakenly used to mean the same as the
British Indian Empire , which included both the provinces and the Native States . Jump up ^ `` Imperial Impressions '' . Hindustan Times . 20 July 2011 .
</item 4>

<item 5>
Governor-General of India
Governor - General of India - Wikipedia Governor - General of India Jump to : navigation , search This article is about the position of Governor - General of
India . For individual Governor - Generals , see List of governors - general of India . Viceroy and Governor - General of India Standard of the Governor -
General Louis Mountbatten , the last Viceroy of India & the first Governor - General during the dominion period Style His Excellency Residence Viceroy 's House
Appointer East India Company ( to 1858 ) Monarch of India ( from 1858 ) Formation 20 October 1774 First holder Warren Hastings Final holder Chakravarthi
Rajagopalachari Abolished 26 January 1950 The Governor - General of India ( or , from 1858 to 1947 , officially the Viceroy and Governor - General of India ,
commonly shortened to Viceroy of India ) was originally the head of the British administration in India and , later , after Indian independence in 1947 , the
representative of the Indian head of state . The office was created in 1773 , with the title of Governor - General of the Presidency of Fort William .
</item 5>

<item 6>
Governor-General of India
Most Governors - General and Viceroys were peers . Frequently , a Viceroy who was already a peer would be granted a peerage of higher rank , as with the
granting of a marquessate to Lord Reading and an earldom and later a marquessate to Freeman Freeman - Thomas . Of those Viceroys who were not peers , Sir John
Shore was a baronet , and Lord William Bentinck was entitled to the courtesy title ' Lord ' because he was the son of a Duke . Only the first and last Governors
- General -- Warren Hastings and Chakravarti Rajagopalachari -- as well as some provisional Governors - General , had no honorific titles at all . Flag ( edit )
Flag of the Governor General of India ( 1947 -- 50 ) From around 1885 , the Viceroy of India was allowed to fly a Union Flag augmented in the centre with the '
Star of India ' surmounted by a Crown . This flag was not the Viceroy 's personal flag ; it was also used by Governors , Lieutenant Governors , Chief
Commissioners and other British officers in India .
</item 6>

<item 7>
Governor-General of India
The first Governor - General of British India was Warren Hastings , and the first Governor - General of independent India was Louis Mountbatten . Contents (
hide ) 1 History 2 Functions 3 Council 4 Style and title 5 Flag 6 Residence 7 Insignia 8 See also 9 References 10 Further reading History ( edit ) Warren
Hastings , the first Governor - General of Fort William from 1773 to 1785 . Many parts of the Indian subcontinent were governed by the East India Company ,
which nominally acted as the agent of the Mughal Emperor . In 1773 , motivated by corruption in the Company , the British government assumed partial control
over the governance of India with the passage of the Regulating Act of 1773 . A Governor - General and Supreme Council of Bengal were appointed to rule over the
Presidency of Fort William in Bengal . The first Governor - General and Council were named in the Act . The Charter Act 1833 replaced the Governor - General and
Council of Fort William with the Governor - General and Council of India .
</item 7>

<item 8>
Governor-General of India
Lord Mountbatten addressing the Chamber of Princes as Crown Representative in the 1940s The Governor - General originally had power only over the Presidency of
Fort William in Bengal . The Regulating Act , however , granted them additional powers relating to foreign affairs and defence . The other Presidencies of the
East India Company ( Madras , Bombay and Bencoolen ) were not allowed to declare war on or make peace with an Indian prince without receiving the prior approval
of the Governor - General and Council of Fort William . The powers of the Governor - General , in respect of foreign affairs , were increased by the India Act
1784 . The Act provided that the other Governors under the East India Company could not declare war , make peace or conclude a treaty with an Indian prince
unless expressly directed to do so by the Governor - General or by the Company 's Court of Directors . While the Governor - General thus became the controller
of foreign policy in India , he was not the explicit head of British India . That status came only with the Charter Act 1833 , which granted him ``
superintendence , direction and control of the whole civil and military Government '' of all of British India .
</item 8>

<item 9>
Governor-General of India
The title of viceroy was abandoned when British India split into the two independent dominions of India and Pakistan , but the office of governor - general
continued to exist in each country separately -- until they adopted republican constitutions in 1950 and 1956 , respectively . Until 1858 , the governor -
general was selected by the Court of Directors of the East India Company , to whom he was responsible . Thereafter , he was appointed by the sovereign on the
advice of the British government ; the Secretary of State for India , a member of the UK Cabinet , was responsible for instructing him on the exercise of his
powers . After 1947 , the sovereign continued to appoint the governor - general , but did so on the advice of the Indian government . Governors - General served
at the pleasure of the sovereign , though the practice was to have them serve five - year terms . Governors - General could have their commission rescinded ;
and if one was removed , or left , a provisional governor - general was sometimes appointed until a new holder of the office could be chosen .
</item 9>

<item 10>
Governor-General of India
The officer had direct control only over Fort William , but supervised other British East India Company officials in India . Complete authority over all of
British India was granted in 1833 , and the official came to be known as the `` Governor - General of India '' . In 1858 , the territories of the East India
Company came under the direct control of the British government ; see British Raj . The governor - general ( now also the viceroy ) headed the central
government of India , which administered the provinces of British India , including the Punjab , Bengal , Bombay , Madras , the United Provinces , and others .
However , much of India was not ruled directly by the British government ; outside the provinces of British India , there were hundreds of nominally sovereign
princely states or `` native states '' , whose relationship was not with the British government , but directly with the monarch . To reflect the governor -
general 's role as the representative of the monarch to the feudal rulers of the princely states , from 1858 the term Viceroy and Governor - General of India (
known in short as the Viceroy of India ) was applied to him .
</item 10>
=== END ITEM id=e5b79419ad4f22cf ===

=== ITEM id=ad8dea95881488da ===
Gold sentences and facts:
Sentence 0: Donald Trump is the 45th and current president , inaugurated on January 20 , 2017 .
  Fact 0.0: Donald Trump is the 45th president.
  Fact 0.1: Donald Trump is the current president.
  Fact 0.2: Donald Trump was inaugurated on January 20, 2017.
Sentence 1: President of the United States of America Presidential Seal Presidential Standard Incumbent Donald Trump since January 20 , 2017 Executive Branch of the U.S. Government Executive Office of the President Style Mr. President ( informal ) The Honorable ( formal ) His Excellency ( international correspondence ) Status Head of State Head of Government Member of Cabinet Domestic Policy Council National Economic Council National Security Council Residence White House Seat Washington , D.C. Appointer Electoral College of the United States Term length Four years , renewable once ( Amendment XXII ) Constituting instrument United States Constitution Inaugural holder George Washington Formation March 4 , 1789 ( 229 years ago ) ( 1789 - 03 - 04 ) Salary $400,000 annually Website whitehouse.gov
  Fact 1.0: The office of President of the United States of America is part of the Executive Branch of the U.S. Government.
  Fact 1.1: The incumbent is Donald Trump, since January 20, 2017.
  Fact 1.2: The office is part of the Executive Office of the President.
  Fact 1.3: The style of address is 'Mr. President' (informal), 'The Honorable' (formal), and 'His Excellency' (international correspondence).
  Fact 1.4: The office has the status of Head of State and Head of Government.
  Fact 1.5: The office is a member of the Cabinet, the Domestic Policy Council, the National Economic Council, and the National Security Council.
  Fact 1.6: The official residence is the White House.
  Fact 1.7: The seat is Washington, D.C.
  Fact 1.8: The appointer is the Electoral College of the United States.
  Fact 1.9: The term length is four years, renewable once (Amendment XXII).
  Fact 1.10: The constituting instrument is the United States Constitution.
  Fact 1.11: The inaugural holder is George Washington.
  Fact 1.12: The office was formed on March 4, 1789, which the page describes as 229 years ago.
  Fact 1.13: The salary is $400,000 annually.
  Fact 1.14: The official website is whitehouse.gov.

Retrieved items:
<item 1>
President of the United States
The president is further empowered to grant federal pardons and reprieves , and to convene and adjourn either or both houses of Congress under extraordinary
circumstances . The president directs the foreign and domestic policies of the United States , and takes an active role in promoting his policy priorities to
members of Congress . In addition , as part of the system of checks and balances , Article One of the United States Constitution gives the president the power
to sign or veto federal legislation . Since the office of president was established in 1789 , its power has grown substantially , as has the power of the
federal government as a whole . Through the Electoral College , the registered voters indirectly elect the president and vice president to a four - year term .
This is the only federal election in the United States which is not decided by popular vote . Nine vice presidents became president by virtue of a president 's
intra-term death or resignation . The Twenty - second Amendment precludes any United States citizen from being elected president for a third term .
</item 1>

<item 2>
President of the United States
If this occurs , then the vice president will assume the presidential powers and duties as acting president ; however , the president can declare that no such
inability exists and resume the discharge of the presidential powers and duties . If the vice president and Cabinet contest this claim , it is up to Congress ,
which must meet within two days if not already in session , to decide the merit of the claim . The United States Constitution mentions the resignation of the
president , but does not regulate its form or the conditions for its validity . Pursuant to federal law , the only valid evidence of the president 's
resignation is a written instrument to that effect , signed by the president and delivered to the office of the Secretary of State . This has only occurred once
, when Richard Nixon delivered a letter to Henry Kissinger to that effect . Section 1 of the Twenty - fifth Amendment states that the vice president becomes
president upon the removal from office , death , or resignation of the preceding president .
</item 2>

<item 3>
Graph query row:
person: Donald Trump
connection: President of the United States
relationship: RELATED_TO
term_or_status: Incumbent since January 20, 2017
</item 3>

<item 4>
President of the United States
President of the United States - wikipedia President of the United States Jump to : navigation , search `` POTUS '' redirects here . For the political talk
radio channel , see P.O.T.U.S. ( Sirius XM ) . For other uses , see President of the United States ( disambiguation ) . For a list , see List of Presidents of
the United States . President of the United States of America Presidential Seal Presidential Standard Incumbent Donald Trump since January 20 , 2017 Executive
Branch of the U.S. Government Executive Office of the President Style Mr. President ( informal ) The Honorable ( formal ) His Excellency ( international
correspondence ) Status Head of State Head of Government Member of Cabinet Domestic Policy Council National Economic Council National Security Council Residence
White House Seat Washington , D.C. Appointer Electoral College of the United States Term length Four years , renewable once ( Amendment XXII ) Constituting
instrument United States Constitution Inaugural holder George Washington Formation March 4 , 1789 ( 229 years ago ) ( 1789 - 03 - 04 ) Salary $400,000 annually
Website whitehouse.gov This article is part of a series on the Politics of the United
</item 4>

<item 5>
President of the United States
Book Category Portal Heads of state and government of North America Heads of state Sovereign states Antigua and Barbuda Bahamas Barbados Belize Canada Costa
Rica Cuba Dominica Dominican Republic El Salvador Grenada Guatemala Haiti Honduras Jamaica Mexico Nicaragua Panama Saint Kitts and Nevis Saint Lucia Saint
Vincent and the Grenadines Trinidad and Tobago United States Heads of government Sovereign states Antigua and Barbuda Bahamas Barbados Belize Canada Costa Rica
Cuba Dominica Dominican Republic El Salvador Grenada Guatemala Haiti Honduras Jamaica Mexico Nicaragua Panama Saint Kitts and Nevis Saint Lucia Saint Vincent
and the Grenadines Trinidad and Tobago United States VIAF : 123879793 LCCN : n80001199 ISNI : 0000 0001 2106 1812 GND : 2033432 - 1 SUDOC : 026377519 BNF :
cb11863688j ( data ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=President_of_the_United_States&oldid=837764569 '' Categories : Presidents of
the United States Presidency of the United States United States presidential history 1789 establishments in the United States Hidden categories : CS1 : Julian
-- Gregorian uncertainty
</item 5>

<item 6>
President of the United States
000 annually Website whitehouse.gov This article is part of a series on the Politics of the United States of America Federal Government ( show ) Constitution of
the United States Law Taxation Legislature ( show ) United States Congress House of Representatives Speaker Paul Ryan ( R ) Majority Leader Kevin McCarthy ( R )
Minority Leader Nancy Pelosi ( D ) Congressional districts United States Senate President Mike Pence ( R ) President Pro Tempore Orrin Hatch ( R ) President Pro
Tempore Emeritus Patrick Leahy ( D ) Majority Leader Mitch McConnell ( R ) Minority Leader Chuck Schumer ( D ) Executive ( show ) President of the United States
Donald Trump ( R ) Vice President of the United States Mike Pence ( R ) Cabinet Federal agencies Executive Office Judiciary ( show ) Supreme Court of the United
States Chief Justice John Roberts Kennedy Thomas Ginsburg Breyer Alito Sotomayor Kagan Gorsuch Courts of Appeals District Courts ( list ) Other tribunals
Elections ( show ) Presidential elections Midterm elections Off - year elections Political parties ( show ) Democratic Republican Third parties Federalism (
show ) State Government Governors Legislatures ( List ) State courts Local government Other countries Atlas The President of the United States ( POTUS / ˈpoʊtəs
/
</item 6>

<item 7>
President of the United States
Images of documents written by U.S. presidents Presidents of the United States ( list ) George Washington ( 1789 -- 1797 ) John Adams ( 1797 -- 1801 ) Thomas
Jefferson ( 1801 -- 1809 ) James Madison ( 1809 -- 1817 ) James Monroe ( 1817 -- 1825 ) John Quincy Adams ( 1825 -- 1829 ) Andrew Jackson ( 1829 -- 1837 )
Martin Van Buren ( 1837 -- 1841 ) William Henry Harrison ( 1841 ) John Tyler ( 1841 -- 1845 ) James K. Polk ( 1845 -- 1849 ) Zachary Taylor ( 1849 -- 1850 )
Millard Fillmore ( 1850 -- 1853 ) Franklin Pierce ( 1853 -- 1857 ) James Buchanan ( 1857 -- 1861 ) Abraham Lincoln ( 1861 -- 1865 ) Andrew Johnson ( 1865 --
1869 ) Ulysses S. Grant ( 1869 -- 1877 ) Rutherford B. Hayes ( 1877 -- 1881 ) James A.
</item 7>

<item 8>
President of the United States
The President of the United States ( POTUS / ˈpoʊtəs / POH - təs ) is the head of state and head of government of the United States of America . The president
directs the executive branch of the federal government and is the commander - in - chief of the United States Armed Forces . In contemporary times , the
president is looked upon as the world 's most powerful political figure and as the leader of the only current global superpower . The role includes
responsibility for the world 's most expensive military that has the second largest nuclear arsenal . The president also leads the nation with the largest
economy by nominal GDP . The president possesses significant domestic and international hard and soft power . Article II of the Constitution establishes the
executive branch of the federal government . It vests the executive power of the United States in the president . The power includes the execution and
enforcement of federal law , alongside the responsibility of appointing federal executive , diplomatic , regulatory and judicial officers , and concluding
treaties with foreign powers with the advice and consent of the Senate .
</item 8>

<item 9>
President of the United States
Jump up ^ The informal term POTUS originated in the Phillips Code , a shorthand method created in 1879 by Walter P. Phillips for the rapid transmission of press
reports by telegraph . Jump up ^ The nine vice presidents who succeeded to the presidency upon their predecessor 's death or resignation and finished - out that
unexpired term are John Tyler ( 1841 ) ; Millard Fillmore ( 1850 ) ; Andrew Johnson ( 1865 ) ; Chester A. Arthur ( 1881 ) ; Theodore Roosevelt ( 1901 ) ; Calvin
Coolidge ( 1923 ) ; Harry S. Truman ( 1945 ) ; Lyndon B. Johnson ( 1963 ) ; and Gerald Ford ( 1974 ) . Jump up ^ Foreign - born American citizens who met the
age and residency requirements at the time the Constitution was adopted were also eligible for the presidency . However , this allowance has since become
obsolete . References Jump up ^ `` How To Address The President ; He Is Not Your Excellency Or Your Honor , But Mr. President '' . The New York Times . August 2
, 1891 .
</item 9>

<item 10>
President of the United States
Kennedy ( 1961 -- 1963 ) Lyndon B. Johnson ( 1963 -- 1969 ) Richard Nixon ( 1969 -- 1974 ) Gerald Ford ( 1974 -- 1977 ) Jimmy Carter ( 1977 -- 1981 ) Ronald
Reagan ( 1981 -- 1989 ) George H.W. Bush ( 1989 -- 1993 ) Bill Clinton ( 1993 -- 2001 ) George W. Bush ( 2001 -- 2009 ) Barack Obama ( 2009 -- 2017 ) Donald
Trump ( 2017 -- present ) Presidency timelines Wilson Harding Coolidge Hoover F.D. Roosevelt Truman Eisenhower Kennedy L.B. Johnson Nixon Ford Carter Reagan
G.H.W. Bush Clinton G.W. Bush Obama Trump Book Category Chief executives of the United States Federal President of the United States State governors ( current
list ) Alabama Alaska Arizona Arkansas California list Colorado Connecticut Delaware Florida Georgia Hawaii list Idaho Illinois list Indiana list Iowa list
Kansas Kentucky list Louisiana Maine list Maryland list Massachusetts Michigan list Minnesota list Mississippi Missouri Montana Nebraska list Nevada New
Hampshire list New Jersey list New Mexico list New York list North Carolina list North Dakota list Ohio Oklahoma list Oregon
</item 10>
=== END ITEM id=ad8dea95881488da ===

=== ITEM id=8da4b2cbb96eafd8 ===
Gold sentences and facts:
Sentence 0: The 1836 article and other sources make it clear that this verse was not used soon after 1745 , and certainly before the song became accepted as the British national anthem in the 1780s and 1790s . It was included as an integral part of the song in the Oxford Book of Eighteenth Century Verse of 1926 , although erroneously referencing the `` fourth verse '' to the Gentleman 's Magazine article of 1745 .
  Fact 0.0: The 1836 article and other sources show that this verse was not used soon after 1745.
  Fact 0.1: The 1836 article and other sources show that this verse was certainly not used before the song became accepted as the British national anthem in the 1780s and 1790s.
Sentence 1: The French Marquise de Créquy wrote in her Souvenirs that the words `` Grand Dieu , sauvez le Roi ! '' were indeed composed by Jean - Baptiste Lully in gratitude for the survival by Louis XIV of an anal fistula operation . ( The surgical knife purpose - made for the occasion is on display in the Musée d'histoire de la médecine . ) Allegedly , Lully had set words by Marie de Brinon to music , and Créquy claimed that the tune was later plagiarised by Handel , and so adopted by the British as `` un de leurs air nationaux '' . ( The entire story might have been intended as a joke . ) Translated into Latin under the title Domine , Salvum Fac Regem , it became the French anthem until 1792 . After the Battle of Culloden , the Hanover dynasty supposedly adopted this melody as the British anthem .
  Fact 1.0: The verse was included as an integral part of the song in the Oxford Book of Eighteenth Century Verse of 1926.
  Fact 1.1: The Oxford Book of Eighteenth Century Verse of 1926 erroneously referenced the 'fourth verse' to the Gentleman's Magazine article of 1745.

Retrieved items:
<item 1>
“God Save the Queen,” or “God Save the King” when the monarch is male, is the national or royal anthem of the United Kingdom and numerous Commonwealth realms,
territories, and Crown dependencies. Its tune’s composer is unknown; it may derive from plainchant, though it is sometimes attributed to John Bull. It was
adopted in 1745.
</item 1>

<item 2>
God Save the Queen
Republic . Retrieved 11 November 2017 . External links ( edit ) Wikisource has original text related to this article : God Save the Queen Wikimedia Commons has
media related to God Save the Queen . National Anthem at the Royal Family website Streaming audio , lyrics and information about God Save the Queen Department
of Canadian Heritage -- Royal anthem page God Save Great George our King : -- article discussing different versions of the lyrics Free sheet music of God Save
the Queen from Cantorion.org A Point of View : Is it time for a new British national anthem ? BBC NEWS . Published 15 January 2016 . Retrieved 17 July 2017 .
National anthems of Europe Sovereign states Albania Andorra Armenia Austria Azerbaijan Belarus Belgium Bosnia and Herzegovina Bulgaria Croatia Cyprus Czech
Republic Denmark Estonia Finland France Georgia Germany Greece Hungary Iceland Ireland Italy Kazakhstan Latvia Liechtenstein Lithuania Luxembourg Macedonia
Malta Moldova Monaco Montenegro Netherlands Norway Poland Portugal Romania Russia San Marino Serbia Slovakia Slovenia Spain Sweden Switzerland Turkey Ukraine
United Kingdom Vatican City Territories , dependencies , other areas United Kingdom Constituent countries England Northern Ireland Scotland Wales British
Overseas Territories Gibraltar Crown dependencies Guernsey Isle of Man Jersey Counties Cornwall Lincolnshire Yorkshire Russia Adygea
</item 2>

<item 3>
God Save the Queen
The first music proper , a performance of the National Anthem was programmed by Christopher Strachey on the Mark II Manchester Electronic Computer at same venue
, in 1951 . Later that year , short extracts of three pieces , the first being the National Anthem , were recorded there by a BBC outside broadcasting unit :
the other pieces being `` Ba Ba Black Sheep , and `` In the Mood '' . Researchers at the University of Canterbury , Christchurch restored the acetate master
disc in 2016 and the results may be heard on Soundcloud . Reception ( edit ) The philosopher and reformer Jeremy Bentham praised `` God Save the King '' in 1796
: `` the melody recommending itself by beauty to the most polished ears , and by its simplicity to the rudest ear . A song of this complexion , implanted by the
habit of half a century in the mass of popular sentiment , can not be refused a place in the inventory of the national blessings . '' Ludwig van Beethoven wrote
`` I have to show the English a little of what a blessing ' God Save the King ' is '' . Alex Marshall , the British author of Republic or Death !
</item 3>

<item 4>
God Save the Queen
It is the national anthem of the United Kingdom and one of two national anthems used by New Zealand since 1977 , as well as for several of the UK 's territories
that have their own additional local anthem . It is also the royal anthem -- played specifically in the presence of the monarch -- of all the aforementioned
countries , as well as Australia ( since 1984 ) , Canada ( since 1980 ) , Barbados and Tuvalu . In countries not previously part of the British Empire , the
tune of `` God Save the Queen '' has provided the basis for various patriotic songs , though still generally connected with royal ceremony . In the United
States , the melody is used for the patriotic song `` My Country , ' Tis of Thee '' . The melody is also used for the national anthem of Liechtenstein , `` Oben
am jungen Rhein '' . Beyond its first verse , which is consistent , `` God Save the Queen / King '' has many historic and extant versions . Since its first
publication , different verses have been added and taken away and , even today , different publications include various selections of verses in various orders .
In general , only one verse is sung .
</item 4>

<item 5>
“God Save the Queen” is the UK’s national anthem and a royal anthem for several countries, including Australia, Canada, Barbados and Tuvalu. New Zealand has
used it alongside another national anthem since 1977. Its melody inspired the US patriotic song “My Country, ’Tis of Thee,” Liechtenstein’s anthem, and
Iceland’s former de facto anthem. It is also used in hymns and popular culture. The song has many historical versions, though usually only one verse
</item 5>

<item 6>
God Save the Queen
Send him victorious Happy and glorious Long to reign over us God save the king ! Use in the United Kingdom ( edit ) See also : National anthem of England The
phrase `` God Save the King '' in use as a rallying cry to the support of the monarch and the UK 's forces `` God Save the Queen '' is the national anthem of
the United Kingdom of Great Britain and Northern Ireland . Like many aspects of British constitutional life , its official status derives from custom and use ,
not from Royal Proclamation or Act of Parliament . In general only one or two verses are sung , but on rare occasions three . The variation in the UK of the
lyrics to `` God Save the Queen '' is the oldest amongst those currently used , and forms the basis on which all other versions used throughout the Commonwealth
are formed ; though , again , the words have varied throughout these years . England has no official national anthem of its own ; `` God Save the Queen '' is
treated as the English national anthem when England is represented at sporting events ( though there are some exceptions to this rule , such as cricket where
Jerusalem is used ) .
</item 6>

<item 7>
“God Save the Queen” (or “King”) is the UK’s national anthem by custom, not statute, and a royal anthem in several Commonwealth countries. Usually one or two
verses are sung; the first six bars serve as royal and some vice-regal salutes. Its use varies across the UK and Commonwealth: New Zealand shares anthem status
with “God Defend New Zealand,” while Australia designated it the Royal Anthem in 1984. The tune inspired other national songs and many musical adaptations.
</item 7>

<item 8>
God Save the Queen
Since 1953 , the anthem is sometimes preceded by a fanfare composed by Gordon Jacob for the coronation of Queen Elizabeth II . Alternative British versions (
edit ) There have been several attempts to improve the song by rewriting the words . In the nineteenth century there was some lively debate about the national
anthem and , even then , verse two was considered to be slightly offensive . Notably , the question arose over the phrase `` scatter her enemies . '' Some
thought it placed better emphasis on the respective power of Parliament and the Crown to change `` her '' to `` our '' ; others pointed out that the theology
was somewhat dubious and substituted `` thine '' instead . Sydney G.R. Coles wrote a completely new version , as did Canon F.K. Harford . In 1836 , William
Edward Hickson wrote four alternative verses . The first , third , and fourth of these verses are appended to the National Anthem in the English Hymnal ( which
only includes verses one and three of the original lyrics ) .
</item 8>

<item 9>
God Save the Queen
In April 2007 there was an early day motion , number 1319 , to the British Parliament to propose that there should be a separate England anthem : `` That this
House ... believes that all English sporting associations should adopt an appropriate song that English sportsmen and women , and the English public , would
favour when competing as England '' . An amendment ( EDM 1319A3 ) was proposed by Evan Harris that the song `` should have a bit more oomph than God Save The
Queen and should also not involve God . '' For more information see also : National anthem of England National anthem of Scotland Hen Wlad Fy Nhadau Londonderry
Air Use in other Commonwealth countries ( edit ) `` God Save the King / Queen '' was exported around the world via the expansion of the British Empire , serving
as each country 's national anthem . Throughout the Empire 's evolution into the Commonwealth of Nations , the song declined in use in most states which became
independent .
</item 9>

<item 10>
“God Save the Queen” (or “God Save the King,” depending on the monarch) is the United Kingdom’s national anthem. Its status comes from custom and use rather
than legislation or royal proclamation. Usually one or two verses are sung, and the UK lyrics are the oldest version, forming the basis for other Commonwealth
versions. England has no official anthem but generally uses it at sporting events, with exceptions such as cricket, where “Jerusalem” is used.
</item 10>
=== END ITEM id=8da4b2cbb96eafd8 ===
