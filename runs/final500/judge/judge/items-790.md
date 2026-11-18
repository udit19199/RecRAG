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

=== ITEM id=c7f7460c9ff0149b ===
Gold sentences and facts:
Sentence 0: In probability theory , two events are independent , statistically independent , or stochastically independent if the occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does not affect the probability distribution of the other .
  Fact 0.0: In probability theory, two events can be called independent.
  Fact 0.1: In probability theory, two events can be called statistically independent.
  Fact 0.2: In probability theory, two events can be called stochastically independent.
  Fact 0.3: Two events are independent if the occurrence of one does not affect the probability of occurrence of the other.
  Fact 0.4: Two random variables are independent if the realization of one does not affect the probability distribution of the other.

Retrieved items:
<item 1>
Independence (probability theory)
Thus , the occurrence of B does not affect the probability of A , and vice versa . Although the derived expressions may seem more intuitive , they are not the
preferred definition , as the conditional probabilities may be undefined if P ( A ) or P ( B ) are 0 . Furthermore , the preferred definition makes clear by
symmetry that when A is independent of B , B is also independent of A . More than two events ( edit ) A finite set of events ( A i ) i = 1 n ( \ displaystyle \
( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) is pairwise independent if every pair of events is independent -- that is , if and only if for all distinct pairs of
indices m , k , P ( A m ∩ A k ) = P ( A m ) P ( A k ) . ( \ displaystyle \ mathrm ( P ) ( A_ ( m ) \ cap A_ ( k ) ) = \ mathrm ( P ) ( A_ ( m ) ) \ mathrm ( P )
( A_ ( k ) ) . )
</item 1>

<item 2>
Independence (probability theory)
1 Two random variables 1.2. 2 More than two random variables 1.2. 3 Conditional independence 1.3 Independent σ - algebras 2 Properties 2.1 Self - independence
2.2 Expectation and covariance 2.3 Characteristic function 3 Examples 3.1 Rolling dice 3.2 Drawing cards 3.3 Pairwise and mutual independence 3.4 Mutual
independence 4 See also 5 References Definition ( edit ) For events ( edit ) Two events ( edit ) Two events A and B are independent ( often written as A ⊥ B ( \
displaystyle A \ perp B ) or A ⊥ ⊥ B ( \ displaystyle A \ perp \ ! \ ! \ ! \ perp B ) ) if their joint probability equals the product of their probabilities : P
( A ∩ B ) = P ( A ) P ( B ) ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \ mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) ) .
</item 2>

<item 3>
Independence (probability theory)
A finite set of events is mutually independent if every event is independent of any intersection of the other events -- that is , if and only if for every k -
element subset of ( A i ) i = 1 n ( \ displaystyle \ ( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) , P ( ⋂ i = 1 k A i ) = ∏ i = 1 k P ( A i ) . ( \ displaystyle \
mathrm ( P ) \ left ( \ bigcap _ ( i = 1 ) ^ ( k ) A_ ( i ) \ right ) = \ prod _ ( i = 1 ) ^ ( k ) \ mathrm ( P ) ( A_ ( i ) ) . ) This is called the
multiplication rule for independent events . Note that it is not a single condition involving only the product of all the probabilities of all single events (
see below for a counterexample ) ; it must hold true for all subsets of events .
</item 3>

<item 4>
Independence (probability theory)
A ( \ displaystyle ( \ mathcal ( A ) ) ) and B ( \ displaystyle ( \ mathcal ( B ) ) ) are said to be independent if , whenever A ∈ A ( \ displaystyle A \ in ( \
mathcal ( A ) ) ) and B ∈ B ( \ displaystyle B \ in ( \ mathcal ( B ) ) ) , P ( A ∩ B ) = P ( A ) P ( B ) . ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \
mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) . )
</item 4>

<item 5>
Independence (probability theory)
The random variables in the first space are pairwise independent because P ( A B ) = P ( A C ) = 1 / 2 = P ( A ) , P ( B A ) = P ( B C ) = 1 / 2 = P ( B ) , and
P ( C A ) = P ( C B ) = 1 / 4 = P ( C ) ; but the three random variables are not mutually independent . The random variables in the second space are both
pairwise independent and mutually independent . To illustrate the difference , consider conditioning on two events . In the pairwise independent case , although
any one event is independent of each of the other two individually , it is not independent of the intersection of the other two : P ( A B C ) = 4 40 4 40 + 1 40
= 4 5 ≠ P ( A ) ( \ displaystyle \ mathrm ( P ) ( A BC ) = ( \ frac ( \ frac ( 4 ) ( 40 ) ) ( ( \ frac ( 4 ) ( 40 ) ) + ( \ frac ( 1 ) ( 40 ) ) ) ) = ( \ tfrac
( 4 )
</item 5>

<item 6>
Independence (probability theory)
Drawing cards ( edit ) If two cards are drawn with replacement from a deck of cards , the event of drawing a red card on the first trial and that of drawing a
red card on the second trial are independent . By contrast , if two cards are drawn without replacement from a deck of cards , the event of drawing a red card
on the first trial and that of drawing a red card on the second trial are not independent , because a deck that has had a red card removed has proportionately
fewer red cards . Pairwise and mutual independence ( edit ) Pairwise independent , but not mutually independent , events . Mutually independent events .
Consider the two probability spaces shown . In both cases , P ( A ) = P ( B ) = 1 / 2 and P ( C ) = 1 / 4 .
</item 6>

<item 7>
Independence (probability theory)
) and yet no two of the three events are pairwise independent ( and hence the set of events are not mutually independent ) . This example shows that mutual
independence involves requirements on the products of probabilities of all combinations of events , not just the single events as in this example . See also (
edit ) Copula ( statistics ) Independent and identically distributed random variables Mutually exclusive events Subindependence Linear dependence between random
variables Conditional independence Normally distributed and uncorrelated does not imply independent Mean dependence References ( edit ) Jump up ^ Russell ,
Stuart ; Norvig , Peter ( 2002 ) . Artificial Intelligence : A Modern Approach . Prentice Hall . p. 478 . ISBN 0 - 13 - 790395 - 2 . ^ Jump up to : Feller , W (
1971 ) . `` Stochastic Independence '' . An Introduction to Probability Theory and Its Applications . Wiley . Jump up ^ George , Glyn , `` Testing for the
independence of three events , '' Mathematical Gazette 88 , November 2004 , 568 .
</item 7>

<item 8>
Independence (probability theory)
For more than two events , a mutually independent set of events is ( by definition ) pairwise independent ; but the converse is not necessarily true ( see below
for a counterexample ) . For random variables ( edit ) Two random variables ( edit ) Two random variables X and Y are independent if and only if ( iff ) the
elements of the π - system generated by them are independent ; that is to say , for every a and b , the events ( X ≤ a ) and ( Y ≤ b ) are independent events (
as defined above ) .
</item 8>

<item 9>
Independence (probability theory)
Independence ( probability theory ) - wikipedia Independence ( probability theory ) Jump to : navigation , search Part of a series on Statistics Probability
theory Probability axioms Probability space Sample space Elementary event Event Random variable Probability measure Complementary event Joint probability
Marginal probability Conditional probability Independence Conditional independence Law of total probability Law of large numbers Bayes ' theorem Boole 's
inequality Venn diagram Tree diagram In probability theory , two events are independent , statistically independent , or stochastically independent if the
occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does
not affect the probability distribution of the other . The concept of independence extends to dealing with collections of more than two events or random
variables , in which case the events are pairwise independent if each pair are independent of each other , and the events are mutually independent if each event
is independent of each other combination of events . Contents ( hide ) 1 Definition 1.1 For events 1.1. 1 Two events 1.1. 2 More than two events 1.2 For random
variables 1.2. 1 Two random variables 1.2.
</item 9>

<item 10>
Independence (probability theory)
More than two random variables ( edit ) A set of random variables is pairwise independent if and only if every pair of random variables is independent . Even if
the set of random variables is pairwise independent , it is not necessarily mutually independent as defined next . A set of random variables is mutually
independent if and only if for any finite subset X 1 , ... , X n ( \ displaystyle X_ ( 1 ) , \ ldots , X_ ( n ) ) and any finite sequence of numbers a 1 , ... ,
a n ( \ displaystyle a_ ( 1 ) , \ ldots , a_ ( n ) ) , the events ( X 1 ≤ a 1 ) , ... , ( X n ≤ a n ) ( \ displaystyle \ ( X_ ( 1 ) \ leq a_ ( 1 ) \ ) , \ ldots
, \ ( X_ ( n ) \ leq a_ ( n ) \ ) ) are mutually independent events ( as defined above ) . The measure - theoretically inclined may prefer to substitute events
( X ∈ A ) for events ( X ≤ a ) in the above definition , where A is any Borel set .
</item 10>
=== END ITEM id=c7f7460c9ff0149b ===

=== ITEM id=d3ccea466f9a9aeb ===
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
Many of the species of fish that lived in the sea have been killed off by the combination of pollutants , salt levels , and algal blooms . Dead fish have been
known to wash up in mass quantities on the beaches . The smell of the lake , combined with the stench of the decaying fish , also contributed to the decline of
the tourist industry around the Salton Sea . Many people now visit the Salton Sea and the surrounding settlements to explore the abandoned structures . The town
of Niland is 1.5 miles ( 3 km ) southeast of the sea , with a population of 1,006 . Evidence of geothermal activity is also visible . Mudpots and mud volcanoes
are found on the eastern side of the Salton Sea . A number of geothermal electricity generation plants are located along the southeastern shore of the Salton
Sea in Imperial County . The US Geological Survey describes the smell as `` objectionable '' , `` noxious '' , `` unique '' , and `` pervasive '' .
</item 7>

<item 8>
Salton Sea
San Diego County Water Authority . Retrieved 2014 - 10 - 14 . Jump up ^ See http://www.usbr.gov/lc/region/programs/SaltonSea_2004-2014.xlsx , linked at
https://www2.usgs.gov/saltonsea/monitoring.html Jump up ^ Khaled M. Bali ( 5 October 2013 ) . `` Salton Sea and Salinity '' . UC Davis , Cooperative Extension
Imperial County . Retrieved 2015 - 06 - 29 . The salinity of the Pacific Ocean is approximately 35,000 mg / l ( 1 ppm = 1 mg / l ) . Colorado River water
salinity is about 650 -- 700 mg / L. Salton Sea salinity is about 44,000 mg / L , that is approximately 4.4 % salt . The amount of salts that is deposited in
the Imperial Valley agricultural land with irrigation water is approximately four million tons of salts annually . Jump up ^ Alles , DL ( 2007 - 08 - 06 ) . ``
Geology of the Salton Trough '' ( PDF ) .
</item 8>

<item 9>
Salton Sea
Salton Sea - wikipedia Salton Sea Jump to : navigation , search For other uses , see Salton Sea ( disambiguation ) . Salton Sea Location Colorado Desert
Imperial and Riverside Counties , California , U.S. Coordinates 33 ° 18 ′ N 115 ° 48 ′ W ﻿ / ﻿ 33.3 ° N 115.8 ° W ﻿ / 33.3 ; - 115.8 Coordinates : 33 ° 18 ′ N
115 ° 48 ′ W ﻿ / ﻿ 33.3 ° N 115.8 ° W ﻿ / 33.3 ; - 115.8 Type Endorheic rift lake Primary inflows Alamo River New River Whitewater River Primary outflows None
Catchment area 8,360 square miles ( 21,700 km ) Basin countries United States , Mexico Surface area 343 sq mi ( 889 km ) Max .
</item 9>

<item 10>
Salton Sea
depth 43 ft ( 13 m ) Water volume 6,000,000 acre ⋅ ft ( 7.4 km ) Surface elevation − 236 ft ( − 71.9 m ) ( below sea level ) Settlements Bombay Beach , Desert
Beach , Desert Shores , Salton City , Salton Sea Beach , North Shore References U.S. Geological Survey Geographic Names Information System : Salton Sea Map of
the Salton Sea drainage area The Salton Sea is a shallow , saline , endorheic rift lake located directly on the San Andreas Fault , predominantly in California
's Imperial and Coachella valleys . The lake occupies the lowest elevations of the Salton Sink in the Colorado Desert of Imperial and Riverside Counties in
Southern California . Its surface is 236.0 ft ( 71.9 m ) below sea level as of January 2018 . The deepest point of the sea is 5 ft ( 1.5 m ) higher than the
lowest point of Death Valley . The sea is fed by the New , Whitewater , and Alamo rivers , as well as agricultural runoff , drainage systems , and creeks .
</item 10>
=== END ITEM id=d3ccea466f9a9aeb ===

=== ITEM id=7097b77efb6d288f ===
Gold sentences and facts:
Sentence 0: `` Like a Boss '' is a song written and recorded by comedy hip hop troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean version that was produced in both audio and video . In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work .
  Fact 0.0: Like a Boss is a song.
  Fact 0.1: Like a Boss was written and recorded by the comedy hip hop troupe The Lonely Island.
  Fact 0.2: The Lonely Island recorded Like a Boss for their debut studio album Incredibad.
  Fact 0.3: Like a Boss is a parody of the first official single from Slim Thug's first album Already Platinum.
  Fact 0.4: The first official single from Slim Thug's first album Already Platinum is also named Like a Boss.
  Fact 0.5: Like a Boss had multiple versions.
  Fact 0.6: One version of Like a Boss was audio-only and did not have the celebrity interviewer.
  Fact 0.7: Like a Boss had a clean version that was produced in both audio and video.
  Fact 0.8: In 2011 Like a Boss was subject to a lawsuit.
  Fact 0.9: The 2011 lawsuit over Like a Boss was from the producers of the song.
  Fact 0.10: The producers of Like a Boss claimed they never received proper compensation for their work.

Retrieved items:
<item 1>
Like a Boss
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
Like a Boss
Abby Elliott and Bobby Moynihan can be seen in the video as background dancers . Popularity ( edit ) Prior to the release of the video on Saturday Night Live ,
the term `` Like a Boss '' was not often searched for or used in social context . Though it was indeed searched for prior to the release of the video , the term
clearly spiked at release and peaked on Google in April 2009 , hitting a 100 on the Google Trend chart . It has also become popular on social media sites such
as Twitter Tumblr , and Facebook , where it is a popular hashtag to cite a task completed , goal achieved or successful action that someone is proud of doing .
It has also led to parody videos such as `` Like a Bus '' , where a bus is shown flying through traffic and often bad weather conditions with no regard or
slowing for the bad conditions . On 22 October 2012 , `` Like A Boss '' surpassed 100 million views . The video was featured by The Huffington Post , and TV
Guide . Lawsuit ( edit ) In November 2011 a production team from St. Louis filed charges against Andy Samberg and Co. , NBC Universal and Universal Republic
Records .
</item 2>

<item 3>
Like a Boss
Louis filed charges against Andy Samberg and Co. , NBC Universal and Universal Republic Records . At the root of the suit was the claim that Monique `` Foxx
StarStrukk '' Hines and Aleric `` Rick Tha Rular '' Banks submitted the audio track for `` Like a Boss '' along with the audio track that would become another
hit for The Lonely Island , `` Shy Ronnie '' . The duo claimed that , through a mutual connection in 2008 , they knew Samberg and submitted their tracks for
inclusion in the upcoming debut album , Incredibad . The Lonely Island then recorded their own lyrics on top of the audio track to create the songs . The duo
from StarStrukk productions claimed that they were never compensated properly for their work , both in money and recognition / notoriety . Banks claimed that
originally he signed an agreement giving him a 50 % ownership of the song , given his position as writer and producer . According to his lawsuit , he received a
response from Universal Music Group that gave him only a 25 % interest in `` Like a Boss . ''
</item 3>

<item 4>
Like a Boss
Banks , however , claims that he has yet to see any sort of a royalty for the song . Another track from the duo was later featured on the band 's 2011 album
Turtleneck & Chain in the song `` Shy Ronnie 2 : Ronnie & Clyde . '' Censorship ( edit ) This section does not cite any sources . Please help improve this
section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2015 ) ( Learn how and when to remove this
template message ) On the NBC version of this digital short , the part where Andy has a gun in his mouth and whines , `` Oh fuck , man , I ca n't fuckin ' do it
. Shit ! '' had the shot of Andy with the gun in his mouth shortened and redubbed the line to , `` Oh crap , man , I ca n't effin ' do it . Ugh ! ! ''
References ( edit ) Jump up ^ Van Dril , Eric ( 5 April 2009 ) . `` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' .
</item 4>

<item 5>
Like a Boss
Like a Boss - wikipedia Like a Boss Jump to : navigation , search ( hide ) This article has multiple issues . Please help improve it or discuss these issues on
the talk page . ( Learn how and when to remove these template messages ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page . ( January 2015 ) This article needs additional citations for verification . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed .
</item 5>

<item 6>
Like a Boss
The video ends with the giant fish delivering the final words `` like a boss . '' Video ( edit ) The video features Seth Rogen , recording over the vocals done
by Akiva Schaffer on the single. The video adds additional visuals not included with the words of the song . It ends with Samberg 's tombstone being shown (
which has the epitaph `` He was da Best '' ) , and a skeptical Rogen trying to confirm some of the more bizarre events with Samberg . The `` bomb the Russians
'' part shows Samberg attacking St. Basil 's Cathedral . The video features brief cameo appearances from Saturday Night Live writer John Mulaney as the worker
that Samberg micromanages , Casey Wilson as the woman giving Samberg the memo , Kristen Wiig as Samberg 's office crush Debra , Jason Sudeikis as Samberg 's
boss and Bill Hader as the man on whom Samberg performs fellatio . It also features Jorma Taccone as the man giving him the lawsuit , and Akiva Schaffer selling
him the gun . Abby Elliott and Bobby Moynihan can be seen in the video as background dancers .
</item 6>

<item 7>
Like a Boss
In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work . Contents ( hide ) 1
Song 2 Video 3 Popularity 4 Lawsuit 5 Censorship 6 References 7 External links Song ( edit ) `` Like a Boss '' premiered as a Saturday Night Live Digital Short
on April 4 , 2009 . In the song , Andy Samberg is a businessman , evidently in middle management , having a performance review . The reviewer asks him to
describe the things he does in an average day , and does so for the remainder of the song in his trademark aggressive style while repeating the lyric `` like a
boss '' after every activity . He first lists activities which are fairly mundane , such as communicating to bigger bosses in corporate headquarters , approving
memorandums , remembering birthdays , leading a workshop , and promoting synergy . However , the song takes a foul turn after he mentions his female subordinate
Deborah , whom he repeatedly pursues , rejecting him , causing him to `` swallow sadness '' , face a sexual harassment lawsuit and be passed up for a promotion
.
</item 7>

<item 8>
Like a Boss
Unsourced material may be challenged and removed . ( February 2011 ) ( Learn how and when to remove this template message ) ( Learn how and when to remove this
template message ) `` Like a Boss '' Song by The Lonely Island featuring Seth Rogen from the album Incredibad Recorded 2008 Genre Comedy hip hop Length 1 : 46
Label Universal Republic Records Songwriter ( s ) Andy Samberg , Akiva Schaffer , Jorma Taccone , A. Banks Producer ( s ) Rick tha Rular Incredibad track
listing `` Shrooms ( skit ) '' ( 9 ) `` Like a Boss '' ( 10 ) `` We Like Sportz '' ( 11 ) `` Like a Boss '' is a song written and recorded by comedy hip hop
troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already
Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean
version that was produced in both audio and video .
</item 8>

<item 9>
Like a Boss
`` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' . The Ripcord . Archived from the original on 11 April 2009 . Retrieved 2014 - 06 - 21 . Jump up ^
`` Google Trends - websearch interest : like a boss - Worldwide , 2004 - present '' . Google Trends . Jump up ^ `` Twitter / search - # likeaboss '' .
Twitter.com . Jump up ^ `` like a boss on Tumblr '' . Tumblr.com . Jump up ^ `` # likeaboss '' . Facebook.com . Jump up ^ `` Like a BUS ! Playlist '' .
Youtube.com . Jump up ^ `` Like A Boss : SNL Digital Short ( VIDEO ) '' . The Huffington Post . 6 May 2009 . Retrieved 2014 - 06 - 23 . Jump up ^ DiNunno , Gina
. `` Today 's News : Our Take - Seth Rogen and Andy Samberg in SNL Digital Short `` Like a Boss '' `` . TV Guide.com . Retrieved 23 June 2014 .
</item 9>

<item 10>
Like a Boss
TV Guide.com . Retrieved 23 June 2014 . Jump up ^ Kaufman , Gill . `` ' SNL , ' Andy Samberg Sued By Producers over ' Like A Boss ' - MTV '' . MTV News .
Retrieved 11 July 2014 . External links ( edit ) Beard , Lanford ( November 8 , 2011 ) . `` NBC , Andy Samberg sued for copyright infringement '' .
Entertainment Weekly . Retrieved November 8 , 2011 . Original video : `` Like a Boss '' on YouTube Censored video : `` Like a Boss - Clean Version '' on YouTube
Video and Song Lyrics : `` Like a Boss '' Lonely Island 's Song Page : `` Like a Boss '' ( hide ) The Lonely Island Andy Samberg Akiva Schaffer Jorma Taccone
Discography Studio albums Incredibad Turtleneck & Chain The Wack Album Popstar : Never Stop Never Stopping Singles `` Dick in a Box '' `` Jizz in My Pants '' ``
I 'm on a Boat '' `` I Just Had Sex '' `` The Creep '' `` We 're Back ! ''
</item 10>
=== END ITEM id=7097b77efb6d288f ===

=== ITEM id=0f2e2f2aa3ad54ef ===
Gold sentences and facts:
Sentence 0: A defensive three - second violation , also known as illegal defense , is a basketball rules infraction in the National Basketball Association ( NBA ) introduced in the 2001 -- 2002 season . It is assessed when a member of the defending team spends more than three seconds in the free throw lane ( also known as the 16 - foot lane , or colloquially as `` in the paint '' ) while not actively guarding an opponent . To be considered actively guarding , a defender must be within arm 's length of an opponent and in a guarding position . A three - second count is suspended if :
  Fact 0.0: A defensive three-second violation is also known as illegal defense.
  Fact 0.1: A defensive three-second violation is a basketball rules infraction in the National Basketball Association (NBA).
  Fact 0.2: The defensive three-second rule was introduced in the 2001-2002 season.
  Fact 0.3: A defensive three-second violation is assessed when a member of the defending team spends more than three seconds in the free throw lane while not actively guarding an opponent.
  Fact 0.4: The free throw lane is also known as the 16-foot lane.
  Fact 0.5: The free throw lane is colloquially known as 'in the paint'.
  Fact 0.6: To be considered actively guarding, a defender must be within arm's length of an opponent.
  Fact 0.7: To be considered actively guarding, a defender must be in a guarding position.
  Fact 0.8: A three-second count is suspended under conditions listed after the word 'if'.

Retrieved items:
<item 1>
Defensive three-second violation
`` PBA get fresh look this season '' . Arab News . Retrieved January 13 , 2014 . Jump up ^ `` WNBA Announces Rules Changes for 2013 Season '' . WNBA . Retrieved
Dec 16 , 2012 . Basketball ( outline ) General topics History of basketball James Naismith Variations of basketball Leagues Statistics Glossary of terms Index
of articles Rules Bonus Jump ball Officials Turnover Violation 3 seconds Offense Defense 5 seconds Basket interference Carrying Double dribble Goaltending Shot
clock Traveling Foul Flagrant Personal Technical Game play Air ball Alley - oop Assist Backboard shattering Ball hog Block Buzzer beater Cherry picking Dribble
Crossover Dunk Euro step Fadeaway Fast break Fly Flop Jump shot Layup Finger roll Field goal Four - point play Free throw Hook shot Moves Pick and roll
Positions Posterized Playbook Rebound Point Screen Back screen Slashing Steal Three - pointer Three - point play Uncontested shot Strategy General Sixth man
Offense Continuity offense Flex Shuffle UCLA High Post Wheel Dribble drive motion Four corners Motion Princeton Run and gun Grinnell System Nellie Ball Small
ball Triangle Defense 1
</item 1>

<item 2>
Defensive three-second violation
corners Motion Princeton Run and gun Grinnell System Nellie Ball Small ball Triangle Defense 1 - 2 - 1 - 1 1 -- 3 -- 1 2 -- 3 zone Amoeba Box - and - one Double
team Full - court press Hack - a-Shaq Jordan Rules Line Man - to - man Match - up zone Triangle - and - two Zone Equipment Backboard Ball Breakaway rim Court
Half court Key Net Possession arrow Whistle Clothing Basketball sleeve Finger sleeve Air Jordan Chuck Taylor All - Stars Miscellaneous 100 - point scorers 50 --
40 -- 90 club AAU All - Americans AAU champions Mikan Drill Player tracking Shootaround Tip drill Winning streaks This basketball - related article is a stub .
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Defensive_three-second_violation&oldid=803130800 ''
Categories : Basketball penalties Basketball stubs Hidden categories : Wikipedia articles needing clarification from November 2015 All stub articles Talk About
Wikipedia Français Edit links This page was last edited on 30 September 2017 , at 17 : 04 . About Wikipedia
</item 2>

<item 3>
2017–18 NBA season
Informally dubbed as the `` Zaza Pachulia rule '' . Also , officials can now assess if a player is in a shooting motion when determining if a perimeter foul is
a shooting or a common foul . The type of foul will be determined on the sequencing of the player 's movement if the foul happens before or after the player
starts a shooting motion . Additionally , they can also assess an offensive foul if an offensive player intentionally lunges into the defender or locks arms to
create contact , a rule informally dubbed as the `` Harden rule '' . On September 28 , 2017 , the NBA board of governors approved the changes to the draft
lottery system to discourage teams from losing on purpose . Starting with the 2018 -- 19 NBA season , the teams with the three worst records will have equal
odds at the # 1 pick . In addition , the revised draft lottery will consist of selecting the top four teams instead of the usual three that occurred for over
two decades . The approval for the change was nearly unanimous , with only the Oklahoma City Thunder voting against the change and the Dallas Mavericks
abstaining from voting .
</item 3>

<item 4>
2017–18 NBA season
NBA.com . Turner Sports Interactive , Inc . February 13 , 2016 . Retrieved June 21 , 2017 . Jump up ^ `` A first look at the NBA 's refreshed logo '' . NBA.com.
6 July 2017 . Retrieved 12 July 2017 . Jump up ^ Feldman , Dan . `` NBA moves up trade deadline , reduces timeouts '' . sports.yahoo.com . Yahoo ! Sports .
Retrieved 12 July 2017 . Jump up ^ Ward - Henninger , Colin . `` NBA moves trade deadline to before All - Star Game , reduces amount of timeouts '' .
CBSSports.com . CBS Sports . Retrieved 12 July 2017 . Jump up ^ Brown , Maury . `` NBA Makes Sweeping Pace - Of - Play Changes ; MLB Likely Next '' . Forbes.com
. Forbes . Retrieved 12 July 2017 . Jump up ^ Dauster , Rob ( April 27 , 2017 ) . `` D - League salaries , two - way contracts increase NBA Draft early entries
'' .
</item 4>

<item 5>
2017–18 NBA season
`` D - League salaries , two - way contracts increase NBA Draft early entries '' . CollegeBasketballTalk . Retrieved September 20 , 2017 . Jump up ^ Mahoney ,
Brian . `` NBA referees can rule closeouts as flagrants or technical fouls '' . NBA.com . Retrieved September 22 , 2017 . Jump up ^ Bushnell , Henry . `` The
NBA is cracking down on dangerous Zaza-esque closeouts '' . sports.yahoo.com . Retrieved September 22 , 2017 . Jump up ^ Boone , Kyle ( September 22 , 2017 ) .
`` The NBA is finally cracking down on James Harden 's foul - drawing antics '' . CBSSports.com . Retrieved October 4 , 2017 . Jump up ^ `` NBA implementing '
Zaza Pachulia , ' ' James Harden ' rules '' . NBCSports.com . September 21 , 2017 . Retrieved October 4 , 2017 . Jump up ^ `` NBA Board Of Governors Approves
Changes To Draft Lottery System '' . NBA.com .
</item 5>

<item 6>
Defensive three-second violation
The offense receives one free throw and retains possession of the ball . Prior to the 2001 -- 2002 NBA season , any form of zone defense was considered an
illegal defense violation and resulted in a warning on the first violation and then a technical foul for any subsequent violations . However , the defensive
three - second violation makes it difficult for NBA defenses to play zone , since zone defenses usually position a player in the middle of the key to stop
penetration . The Philippine Basketball Association used to follow the illegal defense rule until the 2003 PBA season when it was abolished . The high school
game does not use this rule , nor does European basketball . The WNBA started using it in 2013 .
</item 6>

<item 7>
2017–18 NBA season
The previous record of 206 three - pointers was set by Eric Gordon last year . For the first time since the 1996 - 97 NBA season , two teams played their last
game against each other for the 8th and final spot in the playoffs . The Minnesota Timberwolves defeated the Denver Nuggets 112 - 106 in overtime to clinch the
final playoff seed in the west . This also ended Minnesota 's 13 - year drought without a playoff appearance having last played in 2003 - 04 season . On April 6
, 2018 , LeBron James became youngest player to score 31,000 career points at 33 years and 97 days and the 7th player in NBA history to do so . He achieved the
record by finishing 44 points , 11 rebounds , and 11 assists in a losing effort against Philadelphia 76ers , 132 - 130 . On April 11 , 2018 , Markelle Fultz
became the youngest player in NBA history to record a triple - double at 19 years and 317 days old .
</item 7>

<item 8>
Defensive three-second violation
Defensive three - second violation - Wikipedia Defensive three - second violation Jump to : navigation , search A defensive three - second violation , also
known as illegal defense , is a basketball rules infraction in the National Basketball Association ( NBA ) introduced in the 2001 -- 2002 season . It is
assessed when a member of the defending team spends more than three seconds in the free throw lane ( also known as the 16 - foot lane , or colloquially as `` in
the paint '' ) while not actively guarding an opponent . To be considered actively guarding , a defender must be within arm 's length of an opponent and in a
guarding position . A three - second count is suspended if : a player is in the act of shooting there is a loss of team control the defender is actively
guarding an opponent the defender clears the painted area it is imminent the defender will become legal In addition , a player guarding an opponent with the
ball may be in the paint without actively guarding the opponent . The team committing a defensive three - second violation is assessed a team technical foul .
The offense receives one free throw and retains possession of the ball .
</item 8>

<item 9>
Defensive three-second violation
The WNBA started using it in 2013 . References ( edit ) Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , B. Retrieved July 25 , 2010 Jump up
^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , D. Retrieved July 25 , 2010 Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , E.
Retrieved July 25 , 2010 Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , Penalty . Retrieved July 25 , 2010 Jump up ^ Blinebury , Fran (
June 8 , 2006 ) . `` Rule changes have NBA back in the fast lane '' . Houston Chronicle . Retrieved July 26 , 2010 . Jump up ^
http://www.sportingcharts.com/dictionary/nba/defensive-three-second-violation.aspx Jump up ^ Cruz , Agnes ( January 24 , 2003 ) . `` PBA get fresh look this
season '' . Arab News .
</item 9>

<item 10>
2017–18 NBA season
The regular season started eight days earlier than the previous season . This move by the NBA aims for fewer `` back - to - back '' games , even fewer than the
previous season which had already reached an all - time low to help address teams that are `` resting '' their superstars . The NBA decreased the maximum number
of timeouts in a regulation game from 18 to 14 . The three team timeouts during the last two minutes of the game were also decreased to two per team in the last
three minutes to improve the flow and pace of the game . The trade deadline was moved from four days after the NBA All - Star Game to ten days before it , so
teams can settle their rosters before the All - Star break . Free - throw shooters will be assessed a delay - of - game violation if they venture outside of the
three - point line in between attempts . The NBA also added two - way contracted players , signifying an increasing partnership between the NBA and the newly
branded NBA G League . Referees can now assess dangerous closeouts of defenders to jump shooters with little space to land for a technical or a flagrant foul .
</item 10>
=== END ITEM id=0f2e2f2aa3ad54ef ===

=== ITEM id=9326e161d87870ad ===
Gold sentences and facts:
Sentence 0: The Uralic languages ( / jʊəˈrælɪk / ; sometimes called Uralian languages / jʊəˈreɪliən / ) form a language family of 38 languages spoken by approximately 25 million people , predominantly in Northern Eurasia . The Uralic languages with the most native speakers are Hungarian , Finnish , and Estonian , which are the official languages of Hungary , Finland , and Estonia , respectively , and of the European Union . Other Uralic languages with significant numbers of speakers are Erzya , Moksha , Mari , Udmurt , and Komi , which are officially recognized languages in various regions of Russia .
  Fact 0.0: The Uralic languages form a language family.
  Fact 0.1: The Uralic languages are sometimes called Uralian languages.
  Fact 0.2: The Uralic language family has 38 languages.
  Fact 0.3: The Uralic languages are spoken by approximately 25 million people.
  Fact 0.4: The Uralic languages are spoken predominantly in Northern Eurasia.
  Fact 0.5: Hungarian, Finnish, and Estonian are the Uralic languages with the most native speakers.
  Fact 0.6: Hungarian is the official language of Hungary.
  Fact 0.7: Finnish is the official language of Finland.
  Fact 0.8: Estonian is the official language of Estonia.
  Fact 0.9: Hungarian, Finnish, and Estonian are official languages of the European Union.
  Fact 0.10: Erzya, Moksha, Mari, Udmurt, and Komi are other Uralic languages with significant numbers of speakers.
  Fact 0.11: Erzya, Moksha, Mari, Udmurt, and Komi are officially recognized languages in various regions of Russia.

Retrieved items:
<item 1>
Uralic languages
These two authors were the first to outline what was to become the classification of the Finno - Ugric , and later Uralic family . This proposal received some
of its initial impetus from the fact that these languages , unlike most of the other languages spoken in Europe , are not part of what is now known as the Indo
- European family . In 1717 , Swedish professor Olof Rudbeck proposed about 100 etymologies connecting Finnish and Hungarian , of which about 40 are still
considered valid . In the same year , the German scholar Johann Georg von Eckhart , in an essay published in Leibniz 's Collectanea Etymologica , proposed for
the first time a relation to the Samoyedic languages . In 1730 , Philip Johan von Strahlenberg published his book Das Nord - und Ostliche Theil von Europa und
Asia ( The North and East Parts of Europe and Asia ) , surveying the geography , peoples and languages of Russia . All the main groups of the Uralic languages
were already identified here . Nonetheless , these relationships were not widely accepted .
</item 1>

<item 2>
Uralic languages
Obsolete or native names are displayed in italics . Finnic ( Fennic , Baltic Finnic , Balto - Finnic , Balto - Fennic ) Hungarian ( Magyar ) Khanty ( Ostyak ,
Handi , Hantõ ) Mansi ( Vogul ) Mari ( Cheremis ) Mordvinic ( Mordvin , Mordvinian ) Permic ( Permian ) Sami ( Saami , Samic , Saamic , Lappic , Lappish )
Samoyedic ( Samoyed ) There is also historical evidence of a number of extinct languages of uncertain affiliation : Merya Muromian Meshcherian ( until 16th
century ? ) Traces of Finno - Ugric substrata , especially in toponymy , in the northern part of European Russia have been proposed as evidence for even more
extinct Uralic languages . Traditional classification ( edit ) All Uralic languages are thought to have descended , through independent processes of language
change , from Proto - Uralic . The internal structure of the Uralic family has been debated since the family was first proposed .
</item 2>

<item 3>
Uralic languages
Nonetheless , these relationships were not widely accepted . Hungarian intellectuals especially were not interested in the theory and preferred to assume
connections with Turkic tribes , an attitude characterized by Merritt Ruhlen as due to `` the wild unfettered Romanticism of the epoch '' . Still , in spite of
this hostile climate , the Hungarian Jesuit János Sajnovics travelled with Maximilian Hell to survey the alleged relationship between Hungarian and Sami .
Sajnovics published his results in 1770 , arguing for a relationship based on several grammatical features . In 1799 , the Hungarian Sámuel Gyarmathi published
the most complete work on Finno - Ugric to that date . Uralic languages in the Russian Empire ( Russian Census of 1897 ; the census was not held in Finland
because it was an autonomous area ) Progress resumed after a number of decades , with the first major field research expeditions on the Uralic languages spoken
in the more eastern parts of Russia .
</item 3>

<item 4>
Uralic languages
The Uralic languages with the most native speakers are Hungarian , Finnish , and Estonian , which are the official languages of Hungary , Finland , and Estonia
, respectively , and of the European Union . Other Uralic languages with significant numbers of speakers are Erzya , Moksha , Mari , Udmurt , and Komi , which
are officially recognized languages in various regions of Russia . The name `` Uralic '' derives from the fact that the areas where the languages are spoken are
found on both sides of the Ural Mountains . Also , the original homeland ( Urheimat ) is commonly hypothesized to be in the vicinity of the Urals . Finno -
Ugric is sometimes used as a synonym for Uralic , though Finno - Ugric is widely understood to exclude the Samoyedic languages . Scholars who do not accept the
traditional notion that Samoyedic split first from the rest of the Uralic family may treat the terms as synonymous .
</item 4>

<item 5>
Uralic languages
Uralic languages - wikipedia Uralic languages Jump to : navigation , search Not to be confused with Urali language . This article needs additional citations for
verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( July 2017 ) (
Learn how and when to remove this template message ) Uralic Uralian languages Geographic distribution Central , Eastern , and Northern Europe , North Asia
Linguistic classification One of the world 's primary language families Proto - language Proto - Uralic Subdivisions Finnic Hungarian Khanty Mansi Mari
Mordvinic Permic Sami Samoyedic ISO 639 - 5 urj Glottolog ural1272 Geographical distribution of the Uralic languages The Uralic languages ( / jʊəˈrælɪk / ;
sometimes called Uralian languages / jʊəˈreɪliən / ) form a language family of 38 languages spoken by approximately 25 million people , predominantly in
Northern Eurasia .
</item 5>

<item 6>
Uralic languages
1 Selected cognates 3.3. 2 Mutual intelligibility 3.4 Comparison 4 Possible relations with other families 4.1 Indo - Uralic 4.2 Uralic -- Yukaghir 4.3 Eskimo --
Uralic 4.4 Uralo - Siberian 4.5 Ural -- Altaic 4.6 Nostratic 4.7 Eurasiatic 4.8 Uralo - Dravidian 4.9 Uralic skepticism 4.10 Other comparisons 5 See also 6
Notes 7 References 7.1 External classification 7.2 Linguistic issues 8 External links 8.1 `` Rebel '' Uralists History ( edit ) Further information : Proto -
Uralic Homeland ( edit ) Further information : Proto - Uralic homeland hypotheses In recent times , linguists often place the Urheimat ( original homeland ) of
the Proto - Uralic language in the vicinity of the Volga River , west of the Urals , close to the Urheimat of the Indo - European languages , or to the east and
southeast of the Urals .
</item 6>

<item 7>
Uralic languages
Classification ( edit ) Relative numbers of speakers of Uralic languages Hungarian 56 % Finnish 20 % Estonian 4.2 % Erzya 2.8 % Moksha 2.5 % Mari 2 % Udmurt 1.9
% Komi 1.6 % Other 8.9 % The Uralic family comprises nine undisputed groups with no consensus classification between them . ( Some of the proposals are listed
in the next section . ) An agnostic approach treats them as separate branches . Obsolete or native names are displayed in italics .
</item 7>

<item 8>
Uralic languages
The internal structure of the Uralic family has been debated since the family was first proposed . Doubts about the validity of most of the proposed higher -
order branchings ( grouping the nine undisputed families ) are becoming more common . A traditional classification of the Uralic languages has existed since the
late 19th century . It has enjoyed frequent adaptation in whole or in part in encyclopedias , handbooks , and overviews of the Uralic family . Donner 's model
is as follows : Ugric ( Ugrian ) Hungarian Ob - Ugric ( Ob Ugrian ) Khanty Mansi Finno - Permic ( Permian - Finnic ) Permic Finno - Volgaic ( Finno - Cheremisic
, Finno - Mari ) Volga - Finnic Mari Mordvinic Finno - Lappic ( Finno - Saamic , Finno - Samic ) Sami Finnic At Donner 's time , the Samoyedic languages were
still poorly known , and he was not able to address their position .
</item 8>

<item 9>
Uralic languages
The Finno - Permic grouping still holds some support , though the arrangement of its subgroups is a matter of some dispute . Mordvinic is commonly seen as
particularly closely related to or part of Finno - Samic . The term Volgaic ( or Volga - Finnic ) was used to denote a branch previously believed to include
Mari , Mordvinic and a number of the extinct languages , but it is now obsolete and considered a geographic classification rather than a linguistic one . Within
Ugric , uniting Mansi with Hungarian rather than Khanty has been a competing hypothesis to Ob - Ugric . Lexical isoglosses ( edit ) Lexicostatistics has been
used in defense of the traditional family tree . A recent re-evaluation of the evidence however fails to find support for Finno - Ugric and Ugric , suggesting
four lexically distinct branches ( Finno - Permic , Hungarian , Ob - Ugric and Samoyedic ) . One alternate proposal for a family tree , with emphasis on the
development of numerals , is as follows : Uralic ( * kektä `` 2 '' ,
</item 9>

<item 10>
Uralic languages
All of these hypotheses are minority views at the present time in Uralic studies . Other comparisons ( edit ) See also : Alternative theories of the Hungarian
language relations Various unorthodox comparisons have been advanced such as Finno - Basque , Hungarian - Etruscan , and Cal - Ugrian . These are considered at
best spurious fringe - theories by specialists . See also ( edit ) Finland portal Hungary portal Languages portal Uralic peoples Finno - Ugric peoples Notes (
edit ) Jump up ^ Hammarström , Harald ; Forkel , Robert ; Haspelmath , Martin , eds. ( 2017 ) . `` Uralic '' . Glottolog 3.0 . Jena , Germany : Max Planck
Institute for the Science of Human History . Jump up ^ Simons , Gary F. ; Fenning , Charles F. `` Uralic '' . Ethnologue . SIL International . Retrieved 14
February 2018 . Jump up ^ Tommola , Hannu ( 2010 ) . `` Finnish among the Finno - Ugrian languages '' . Mood in the Languages of Europe .
</item 10>
=== END ITEM id=9326e161d87870ad ===

=== ITEM id=48024ba459fbb49f ===
Gold sentences and facts:
Sentence 0: Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years elapsed 304 123 47 50
  Fact 0.0: The starting value is 500 million.
  Fact 0.1: The population values, in billions, are 0.5 and 8.
  Fact 0.2: The year values are 1500, 1804, 1927 and 2024.
  Fact 0.3: The years elapsed values are 304, 123, 47 and 50.

Retrieved items:
<item 1>
The table traces world population from under 15 million in 10,000 BC to about 6.5 billion in 2005, with rapid growth accelerating after 1800. Projections show
population rising to roughly 10.4 billion by 2075. Regional shares are expected to shift: Asia’s share declines while Africa’s grows; Europe’s share falls
slightly, and other regions change relatively little. The source data is fragmented and some entries are missing.
</item 1>

<item 2>
World population has generally grown across inhabited continents, though rates vary by region. It rose from about 1.6 billion in 1900 to over 6 billion in
2000—the largest increase in recorded history—and was roughly ten times its 1700 size. Improved sanitation, medical advances, and Green Revolution agriculture
helped reduce mortality and boost growth. In 2000, annual growth was about 1.14%, or 75 million people, down from 88 million in 1989
</item 2>

<item 3>
World population
Milestones by the billions ( edit ) Main article : World population milestones World population milestones in billions Population 5 6 7 8 9 Year 1804 1927 1960
1999 2011 2024 2042 Years elapsed -- -- 123 33 14 13 12 12 13 18 It is estimated that the world population reached one billion for the first time in 1804 . It
was another 123 years before it reached two billion in 1927 , but it took only 33 years to reach three billion in 1960 . Thereafter , the global population
reached four billion in 1974 , five billion in 1987 , six billion in 1999 and , according to the United States Census Bureau , seven billion in March 2012 . The
United Nations , however , estimated that the world population reached seven billion in October 2011 . According to current projections , the global population
will reach eight billion by 2024 , and will likely reach around nine billion by 2042 . Alternative scenarios for 2050 range from a low of 7.4 billion to a high
of more than 10.6 billion .
</item 3>

<item 4>
The global population is unlikely to double again after 2024. Historical doubling times shortened dramatically—from 304 years (1500–1804) to about 50 years
(1974–2024). Resource scarcity may become more common as population grows: Greater Los Angeles, for example, has over 18 million residents despite local water
resources being sufficient for at most one million.
</item 4>

<item 5>
World population reached 1 billion around 1804, 2 billion in 1927, 3 billion in 1960, 4 billion in 1974, 5 billion in 1987, and 6 billion in 1999. Estimates
differ on when it reached 7 billion: the UN says October 2011, while the US Census Bureau says March 2012. Projections put 8 billion around 2024 and 9 billion
around 2042;
</item 5>

<item 6>
World population
The vertical axis is logarithmic and is in millions of people . Population size fluctuates at differing rates in differing regions . Nonetheless , population
growth is the long - standing trend on all inhabited continents , as well as in most individual states . During the 20th century , the global population saw its
greatest increase in known history , rising from about 1.6 billion in 1900 to over 6 billion in 2000 . A number of factors contributed to this increase ,
including the lessening of the mortality rate in many countries by improved sanitation and medical advances , and a massive increase in agricultural
productivity attributed to the Green Revolution . In 2000 , the United Nations estimated that the world 's population was growing at an annual rate of 1.14 % (
equivalent to around 75 million people ) , down from a peak of 88 million per year in 1989 . By 2000 , there were approximately ten times as many people on
Earth as there had been in 1700 .
</item 6>

<item 7>
World population
However , after 2024 , it is unlikely that there will be another doubling of the global population in the 21st century . Historic chart showing the periods of
time the world population has taken to double , from 1700 to 2000 Starting at 500 million Population ( in billions ) 0.5 8 Year 1500 1804 1927 2024 Years
elapsed 304 123 47 50 Starting at 375 million Population ( in billions ) 0.375 0.75 1.5 6 Year 1171 1715 1881 1960 1999 Years elapsed 544 166 79 39
Overpopulation ( edit ) Main articles : Human overpopulation and Malthusian catastrophe Predictions of scarcity ( edit ) Greater Los Angeles lies on a coastal
mediterranean savannah with a small watershed that is able to support at most one million people on its own water ; as of 2015 , the area has a population of
over 18 million . Researchers predict that similar cases of resource scarcity will grow more common as the world population increases .
</item 7>

<item 8>
World population
Projected figures vary depending on underlying statistical assumptions and the variables used in projection calculations , especially the fertility variable .
Long - range predictions to 2150 range from a population decline to 3.2 billion in the `` low scenario '' , to `` high scenarios '' of 24.8 billion . One
extreme scenario predicted a massive increase to 256 billion by 2150 , assuming the global fertility rate remained at its 1995 level of 3.04 children per woman
; however , by 2010 the global fertility rate had declined to 2.52 . There is no estimation for the exact day or month the world 's population surpassed one or
two billion . The points at which it reached three and four billion were not officially noted , but the International Database of the United States Census
Bureau placed them in July 1959 and April 1974 respectively . The United Nations did determine , and commemorate , the `` Day of 5 Billion '' on July 11 , 1987
, and the `` Day of 6 Billion '' on October 12 , 1999 . The Population Division of the United Nations declared the `` Day of 7 Billion '' to be October 31 ,
2011 .
</item 8>

<item 9>
World population—the total number of people alive—was estimated at 7.6 billion in March 2018. It grew continuously after the 1315–17 Great Famine and the Black
Death, when it was about 370 million. Annual growth exceeded 1.8% from 1955 to 1975, peaking at 2.06% in 1965–70. UN projections chart possible population
trends through 2100.
</item 9>

<item 10>
World population
The transition from hyperbolic growth to slower rates of growth is related to the demographic transition . According to the Russian demographer Sergey Kapitsa ,
the world population grew between 67,000 BC and 1965 according to the following formula : N = C τ arccot ⁡ T 0 − T τ ( \ displaystyle N = ( \ frac ( C ) ( \ tau
) ) \ operatorname ( arccot ) ( \ frac ( T_ ( 0 ) - T ) ( \ tau ) ) ) where N is current population T is the current year C = ( 1.86 ± 0.01 ) 10 T = 2007 ± 1 τ
( \ displaystyle \ tau ) = 42 ± 1 Years for world population to double ( edit ) According to linear interpolation and extrapolation of UNDESA population
estimates , the world population has doubled , or will double , in the years listed in the tables below ( with two different starting points ) . During the 2nd
millennium , each doubling took roughly half as long as the previous doubling , fitting the hyperbolic growth model mentioned above .
</item 10>
=== END ITEM id=48024ba459fbb49f ===
