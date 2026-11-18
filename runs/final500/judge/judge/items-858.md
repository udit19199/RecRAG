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

=== ITEM id=00ad5ee69e7fd76f ===
Gold sentences and facts:
Sentence 0: The original start date was January 2002 , but was pushed to February 7 in Los Angeles , California . Locations included Burbank , Downey , New York City , LA / Ontario International Airport ( which doubled for Miami International Airport ) , Quebec City and Montreal . The film was shot in 147 locations in only 52 days . DiCaprio reflected , `` Scenes that we thought would take three days took an afternoon '' . Filming ran from April 25 -- 30 in Park Avenue , just outside the Waldorf - Astoria Hotel . Production moved to Orange , New Jersey and returned to Brooklyn for bank and courthouse scenes . Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport . Quebec City was chosen for its European character and French feel . Place Royale , within Old Quebec , stands for Montrichard -- the church in the background of the arrest scene is Notre - Dame - des - Victoires . Filming ended on May 12 in Montreal .
  Fact 0.0: The original start date was January 2002.
  Fact 0.1: The start date was pushed to February 7 in Los Angeles, California.
  Fact 0.2: Locations included Burbank.
  Fact 0.3: Locations included Downey.
  Fact 0.4: Locations included New York City.
  Fact 0.5: Locations included LA / Ontario International Airport.
  Fact 0.6: LA / Ontario International Airport doubled for Miami International Airport.
  Fact 0.7: Locations included Quebec City.
  Fact 0.8: Locations included Montreal.
  Fact 0.9: The film was shot in 147 locations.
  Fact 0.10: The film was shot in only 52 days.
  Fact 0.11: DiCaprio reflected that scenes thought to take three days took an afternoon.
  Fact 0.12: Filming ran from April 25 to 30 in Park Avenue.
  Fact 0.13: Park Avenue filming was just outside the Waldorf-Astoria Hotel.
  Fact 0.14: Production moved to Orange, New Jersey.
  Fact 0.15: Production returned to Brooklyn for bank and courthouse scenes.
  Fact 0.16: Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport.
  Fact 0.17: Quebec City was chosen for its European character.
  Fact 0.18: Quebec City was chosen for its French feel.
  Fact 0.19: Place Royale, within Old Quebec, stands for Montrichard.
  Fact 0.20: The church in the background of the arrest scene is Notre-Dame-des-Victoires.
  Fact 0.21: Filming ended on May 12 in Montreal.
Sentence 1: Catch Me If You Can is a 2002 American biographical crime film directed and co-produced by Steven Spielberg from a screenplay by Jeff Nathanson . The film is based on the life of Frank Abagnale , who , before his 19th birthday , successfully performed cons worth millions of dollars by posing as a Pan American World Airways pilot , a Georgia doctor and a Louisiana parish prosecutor . His primary crime was check fraud ; he became so experienced that the FBI eventually turned to him for help in catching other checking forgers . The film stars Leonardo DiCaprio and Tom Hanks , with Christopher Walken , Martin Sheen , and Nathalie Baye in supporting roles .
  Fact 1.0: Catch Me If You Can is a 2002 American biographical crime film.
  Fact 1.1: Catch Me If You Can was directed by Steven Spielberg.
  Fact 1.2: Catch Me If You Can was co-produced by Steven Spielberg.
  Fact 1.3: Catch Me If You Can has a screenplay by Jeff Nathanson.
  Fact 1.4: The film is based on the life of Frank Abagnale.
  Fact 1.5: Frank Abagnale successfully performed cons worth millions of dollars before his 19th birthday.
  Fact 1.6: Frank Abagnale performed cons by posing as a Pan American World Airways pilot.
  Fact 1.7: Frank Abagnale performed cons by posing as a Georgia doctor.
  Fact 1.8: Frank Abagnale performed cons by posing as a Louisiana parish prosecutor.
  Fact 1.9: Abagnale's primary crime was check fraud.
  Fact 1.10: Abagnale became so experienced that the FBI eventually turned to him for help.
  Fact 1.11: The FBI turned to Abagnale for help in catching other checking forgers.
  Fact 1.12: The film stars Leonardo DiCaprio.
  Fact 1.13: The film stars Tom Hanks.
  Fact 1.14: Christopher Walken, Martin Sheen, and Nathalie Baye have supporting roles in the film.
Sentence 2: Development for the film started in 1980 , but did not progress until 1997 , when the film rights to Abagnale 's book were sold to Spielberg 's DreamWorks . David Fincher , Gore Verbinski , Lasse Hallström , Miloš Forman , and Cameron Crowe had all been possible candidates for director before Spielberg decided to direct . Filming took place from February to May 2002 . The film was a financial and critical success , and was received positively by both critics and Abagnale .
  Fact 2.0: Development for the film started in 1980.
  Fact 2.1: Development did not progress until 1997.
  Fact 2.2: In 1997, the film rights to Abagnale's book were sold to Spielberg's DreamWorks.
  Fact 2.3: David Fincher was a possible candidate for director before Spielberg decided to direct.
  Fact 2.4: Gore Verbinski was a possible candidate for director before Spielberg decided to direct.
  Fact 2.5: Lasse Hallström was a possible candidate for director before Spielberg decided to direct.
  Fact 2.6: Miloš Forman was a possible candidate for director before Spielberg decided to direct.
  Fact 2.7: Cameron Crowe was a possible candidate for director before Spielberg decided to direct.
  Fact 2.8: Filming took place from February to May 2002.
  Fact 2.9: The film was a financial success.
  Fact 2.10: The film was a critical success.
  Fact 2.11: The film was received positively by both critics and Abagnale.

Retrieved items:
<item 1>
Catch Me If You Can
Catch Me If You Can - wikipedia Catch Me If You Can Jump to : navigation , search This article is about the 2002 film . For other uses , see Catch Me If You Can
( disambiguation ) . Catch Me If You Can Theatrical release poster Directed by Steven Spielberg Produced by Steven Spielberg Walter F. Parkes Screenplay by Jeff
Nathanson Based on Catch Me If You Can by Frank Abagnale and Stan Redding Starring Leonardo DiCaprio Tom Hanks Music by John Williams Cinematography Janusz
Kamiński Edited by Michael Kahn Production company Amblin Entertainment Parkes / MacDonald Productions Distributed by DreamWorks Pictures Release date December
25 , 2002 ( 2002 - 12 - 25 ) Running time 141 minutes Country United States Language English Budget $52 million Box office $352.1 million Catch Me If You Can is
a 2002 American biographical crime film directed and co-produced by Steven Spielberg from a screenplay by Jeff Nathanson .
</item 1>

<item 2>
Catch Me If You Can
Broadway - Aimed `` Catch Me If You Can Ends Seattle Premiere Run Aug. 16 '' Archived August 19 , 2009 , at the Wayback Machine ... Playbill . August 16 , 2009
. Retrieved 2011 - 11 - 17 . Jump up ^ `` CATCH ME IF YOU CAN to Open on Broadway April 10 ; Previews March 7 , 2011 '' . BroadwayWorld.com . Retrieved October
17 , 2011 . Jump up ^ `` Catch Me If You Can Books Broadway 's Neil Simon Theatre '' Archived September 26 , 2013 , at the Wayback Machine ... Playbill .
Retrieved October 17 , 2011 . Jump up ^ `` 2011 Tony Nominations Announced ; Book of Mormon Earns 14 Nominations '' Archived September 14 , 2011 , at the
Wayback Machine ... Playbill . Retrieved October 17 , 2011 . Frank Abagnale , Jr. and Stan Redding .
</item 2>

<item 3>
Catch Me If You Can
The film is based on the life of Frank Abagnale , who , before his 19th birthday , successfully performed cons worth millions of dollars by posing as a Pan
American World Airways pilot , a Georgia doctor and a Louisiana parish prosecutor . His primary crime was check fraud ; he became so experienced that the FBI
eventually turned to him for help in catching other checking forgers . The film stars Leonardo DiCaprio and Tom Hanks , with Christopher Walken , Martin Sheen ,
and Nathalie Baye in supporting roles . Development for the film started in 1980 , but did not progress until 1997 , when the film rights to Abagnale 's book
were sold to Spielberg 's DreamWorks . David Fincher , Gore Verbinski , Lasse Hallström , Miloš Forman , and Cameron Crowe had all been possible candidates for
director before Spielberg decided to direct . Filming took place from February to May 2002 . The film was a financial and critical success , and was received
positively by both critics and Abagnale .
</item 3>

<item 4>
Catch Me If You Can
By April 2000 , David Fincher was attached to direct over the course of a few months , but dropped out in favor of Panic Room . In July 2000 , Leonardo DiCaprio
had entered discussions to star , with Gore Verbinski to direct . Steven Spielberg signed on as producer , and filming was set to begin in March 2001 . Casting
( edit ) Verbinski cast James Gandolfini as Carl Hanratty , Ed Harris as Frank Abagnale , Sr. , and Chloë Sevigny as Brenda Strong . Verbinski dropped out
because of DiCaprio 's commitment on Gangs of New York . Lasse Hallström was in negotiations to direct by May 2001 , but dropped out in July 2001 . At this
stage Harris and Sevigny left the film , but Gandolfini was still attached . Spielberg , co-founder of DreamWorks , offered the job of director to Miloš Forman
, and considered hiring Cameron Crowe .
</item 4>

<item 5>
Catch Me If You Can
The marketing department was careful to market the film as `` inspired by a true story '' in order to avoid controversy similar to that surrounding A Beautiful
Mind ( 2001 ) and The Hurricane ( 1999 ) , both of which deviated from history . The premiere took place at Westwood , Los Angeles , California on December 18 ,
2002 . Box Office ( edit ) Catch Me If You Can was released on December 25 , 2002 , earning slightly above $30 million in 3,225 theaters during its opening
weekend . The film went on to gross $164.6 million in North America and $187.5 million in foreign countries , coming at a worldwide total of $352.1 million .
The film was a financial success , recouping the $52 million budget six times over . Catch Me If You Can was the eleventh highest - grossing film of 2002 .
Minority Report ( also directed by Spielberg ) was tenth highest . Critical response ( edit ) On Rotten Tomatoes , the film has a certified fresh rating of 96 %
, based on 197 reviews , with an average rating of 7.9 / 10 .
</item 5>

<item 6>
Catch Me If You Can
Frank Abagnale , Jr. and Stan Redding . Catch Me If You Can : The Amazing True Story of the Youngest and Most Daring Con Man in the History of Fun and Profit .
( ISBN 0 - 06 - 052971 - 7 ) . External links ( edit ) Wikiquote has quotations related to : Catch Me If You Can Catch Me If You Can on IMDb Catch Me If You Can
at AllMovie Catch Me If You Can at the TCM Movie Database Catch Me If You Can at the American Film Institute Catalog Catch Me If You Can at Box Office Mojo
Catch Me If You Can at Rotten Tomatoes Catch Me If You Can at Metacritic An Interview with Leonardo DiCaprio - Movies Feature at IGN Official website for Catch
Me If You Can the musical ( hide ) Steven Spielberg Filmography Awards and nominations Films directed Firelight ( 1964 ) Slipstream ( 1967 ) Amblin ' ( 1968 )
Night Gallery ( `` Eyes '' segment , 1969 ) L.A.
</item 6>

<item 7>
Catch Me If You Can
British Board of Film Classification . December 13 , 2002 . Retrieved February 13 , 2016 . Jump up ^ Van Luling , Todd ( October 17 , 2014 ) . `` 11 Easter Eggs
You Never Noticed In Your Favorite Movies '' . The Huffington Post . TheHuffingtonPost.com , Inc . Retrieved January 29 , 2015 . ^ Jump up to : Frank Abagnale (
September 3 , 2002 ) . `` Comments '' . Abagnale & Associates . Archived from the original on February 16 , 2009 . Retrieved June 30 , 2008 . ^ Jump up to :
Claude Brodesser ; Dana Harris ( August 21 , 2001 ) . `` D'Works to play ' Catch ' '' . Variety . Retrieved June 30 , 2008 . Jump up ^ Charles Lyons ; Dade
Hayes ( January 8 , 2002 ) . `` D'Works sets play date for pricey ' Catch ' '' . Variety . Retrieved June 29 , 2008 .
</item 7>

<item 8>
Catch Me If You Can
Williams also earned a Grammy Award nomination . Elements of the film were later parodied in The Simpsons episode `` Catch ' Em If You Can '' . Home media (
edit ) Catch Me If You Can was released on DVD on May 6 , 2003 and on Blu - ray on December 4 , 2012 . Musical adaptation ( edit ) Main article : Catch Me If
You Can ( musical ) A musical adaptation of the same name premiered at the 5th Avenue Theatre in Seattle , Washington in July 2009 , starring Aaron Tveit and
Norbert Leo Butz . It began previews on Broadway at the Neil Simon Theatre on March 11 , 2011 and officially opened April 10 , 2011 . The musical was nominated
for four Tony Awards , including Best Musical . See also ( edit ) The Great Impostor , a 1961 film based on the story of an impostor named Ferdinand Waldo
Demara . The Pretender , a TV series References ( edit ) Jump up ^ `` CATCH ME IF YOU CAN ( 12A ) '' . British Board of Film Classification . December 13 , 2002
.
</item 8>

<item 9>
Catch Me If You Can
Locations included Burbank , Downey , New York City , LA / Ontario International Airport ( which doubled for Miami International Airport ) , Quebec City and
Montreal . The film was shot in 147 locations in only 52 days . DiCaprio reflected , `` Scenes that we thought would take three days took an afternoon '' .
Filming ran from April 25 -- 30 in Park Avenue , just outside the Waldorf - Astoria Hotel . Production moved to Orange , New Jersey and returned to Brooklyn for
bank and courthouse scenes . Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport . Quebec City was chosen for its
European character and French feel . Place Royale , within Old Quebec , stands for Montrichard -- the church in the background of the arrest scene is Notre -
Dame - des - Victoires . Filming ended on May 12 in Montreal . Music ( edit ) Main article : Catch Me If You Can ( soundtrack ) The film 's soundtrack was
released on December 10 , 2002 by DreamWorks Records . The original score was composed and conducted by John Williams .
</item 9>

<item 10>
Catch Me If You Can
wikipedia.org/w/index.php?title=Catch_Me_If_You_Can&oldid=821436887 '' Categories : 2002 films English - language films 2000s biographical films 2000s crime
drama films Amblin Entertainment films American biographical films American chase films American crime drama films American films Biographical films about
fraudsters Crime films based on actual events Drama films based on actual events DreamWorks Pictures films Films scored by John Williams Films about con artists
Films about identity theft Films based on biographies Films directed by Steven Spielberg Films set in Atlanta Films set in California Films set in Florida Films
set in France Films set in Louisiana Films set in Westchester County , New York Films set in New York City Films set in the 1960s Films shot in California Films
shot in Montreal Films produced by Steven Spielberg True crime films Fraud in fiction Hidden categories : Webarchive template wayback links Use mdy dates from
October 2015 Good articles Wikipedia articles with GND identifiers Talk Contents About Wikipedia Wikiquote বাংলা Български Català Čeština Dansk Deutsch Español
Euskara فارسی Français Galego 한국어
</item 10>
=== END ITEM id=00ad5ee69e7fd76f ===

=== ITEM id=0ad3a4e41ff7ebc6 ===
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
Additional activities at the base have included experimental testing of solidfuel plane - launched rockets , jet - assist take - off testing , aeroballistic
testing of inert atomic weapon test units at land and marine target areas , training bombing at marine targets , testing of the effects of long - term storage
on atomic weapons , testing of the parachute landing systems of the Project Mercury space capsules , parachute training and testing , and military training
exercises . The base was abandoned in 1978 . The continuing intermittent flooding of the Imperial Valley from the Colorado River led to the idea of the need for
a dam on the Colorado River for flood control . Subsequent evolution ( edit ) A gaseous mud volcano The Salton Sea had some success as a resort area , with
Salton City , Salton Sea Beach , and Desert Shores , on the western shore and Desert Beach , North Shore , and Bombay Beach , built on the eastern shore in the
1950s . However , many of the settlements substantially shrank in size , or have been abandoned , mostly due to the increasing salinity and pollution of the
lake over the years from agricultural runoff and other sources .
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
Originally , it was feared that the waterfall would recede upstream to the true main path of the Colorado , becoming up to 100 to 300 feet ( 30 to 91 m ) high ,
at which point it would be practically impossible to fix the problem . Salton Sea As the basin filled , the town of Salton , a Southern Pacific Railroad siding
, and Torres - Martinez Native American land were submerged . The sudden influx of water and the lack of any drainage from the basin resulted in the formation
of the Salton Sea . The U.S. Navy conducted a preliminary inspection of the Salton Sea in January 1940 , and the Salton Sea Test Base ( SSTB , run by Sandia
Labs ) was initially commissioned as the Naval Auxiliary Air Station , Salton Sea , in October 1942 . The SSTB , just to the south - east of Salton City ,
originally functioned as an operational and training base for seaplanes .
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
Fertilizer runoffs have resulted in eutrophication , with large algal blooms and elevated bacterial levels . By 2014 , large swaths of lake bed were exposed and
salt levels drastically increased due to mandated water transfers to metropolitan areas along the coast and other factors , limiting the water inflow . Besides
the resulting fish kills , the shrinking lake interrupts the bird migration , causes dust clouds , and impacts local tourism negatively . Remediation efforts (
edit ) Past efforts and proposals for a Sea level canal ( edit ) Map of New River and Alamo River courses Alternatives for `` saving '' the Salton Sea have been
evaluated since 1955 . The New River passes from Mexicali , Baja California , to the Imperial Valley , and on to the Salton Sea . Much of the current interest
in the sea was sparked in the 1990s by Congressman Sonny Bono . His widow , Mary Bono Mack , elected to fill his seat , has continued to be interested in the
Salton Sea , as has Representative Jerry Lewis of Redlands . In 1998 , the Sonny Bono Salton Sea Restoration Project was named for the politician .
</item 8>

<item 9>
Salton Sea
Contra Costa Del Norte El Dorado Fresno Glenn Humboldt Imperial Inyo Kern Kings Lake Lassen Los Angeles Madera Marin Mariposa Mendocino Merced Modoc Mono
Monterey Napa Nevada Orange Placer Plumas Riverside Sacramento San Benito San Bernardino San Diego San Francisco San Joaquin San Luis Obispo San Mateo Santa
Barbara Santa Clara Santa Cruz Shasta Sierra Siskiyou Solano Sonoma Stanislaus Sutter Tehama Trinity Tulare Tuolumne Ventura Yolo Yuba Most populous cities Los
Angeles San Diego San Jose San Francisco Fresno Sacramento Long Beach Oakland Bakersfield Anaheim Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Salton_Sea&oldid=836852484 '' Categories : Salton Sea Endorheic lakes of California Lakes of Imperial County ,
California Lakes of Riverside County , California Lakes of the Great Basin Saline lakes Shrunken lakes Coachella Valley Imperial Valley Disasters in California
Engineering failures Environment of California Environmental disasters in the United States Geography of the Colorado Desert Important Bird Areas of California
Natural history of Imperial County , California Tourist attractions in Imperial County ,
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
=== END ITEM id=0ad3a4e41ff7ebc6 ===

=== ITEM id=f17baa273fed3850 ===
Gold sentences and facts:
Sentence 0: ESPN Monday Night Football ( abbreviated as MNF and also known as ESPN Monday Night Football on ABC for rare live special broadcast ) is a live television broadcast of weekly National Football League ( NFL ) games on ESPN in the United States . From 1970 to 2005 , it aired on sister broadcast network ABC . Monday Night Football was , along with Hallmark Hall of Fame and the Walt Disney anthology television series , one of the longest - running prime time programs ever on commercial network television , and one of the highest - rated , particularly among male viewers . MNF is preceded on air by Monday Night Countdown served by Applebee 's .
  Fact 0.0: ESPN Monday Night Football is abbreviated as MNF.
  Fact 0.1: ESPN Monday Night Football is also known as ESPN Monday Night Football on ABC for rare live special broadcast.
  Fact 0.2: ESPN Monday Night Football is a live television broadcast of weekly National Football League (NFL) games.
  Fact 0.3: ESPN Monday Night Football airs on ESPN in the United States.
  Fact 0.4: From 1970 to 2005, Monday Night Football aired on sister broadcast network ABC.
  Fact 0.5: Monday Night Football is one of the longest-running prime time programs ever on commercial network television.
  Fact 0.6: Monday Night Football is, along with Hallmark Hall of Fame and the Walt Disney anthology television series, one of the longest-running prime time programs ever on commercial network television.
  Fact 0.7: Monday Night Football is one of the highest-rated programs, particularly among male viewers.
  Fact 0.8: Monday Night Football is preceded on air by Monday Night Countdown.
  Fact 0.9: Monday Night Countdown is served by Applebee's.
Sentence 1: ESPN Monday Night Football Genre NFL game telecasts Created by Roone Arledge Directed by Chet Forte ( 1970 -- 1986 ) Larry Kamm ( 1987 ) Craig Janoff ( 1988 -- 1999 ) Drew Esocoff ( 2000 -- 2005 ) Chip Dean ( 2006 -- present ) Presented by Commentators : Sean McDonough Lisa Salters Studio hosts : Suzy Kolber Steve Young Randy Moss Matt Hasselbeck Charles Woodson Opening theme `` Score '' by Bob 's Band ( 1970 -- 1975 ) `` All My Rowdy Friends Are Here on Monday Night '' by Hank Williams , Jr. ( 1989 -- 2011 ; 2017 -- present ) `` Heavy Action '' by Johnny Pearson ( 1976 -- 1988 ; 2011 -- 2016 ) ( used as secondary theme from 1989 to 2011 and again from 2017 -- present ) Country of origin United States Original language ( s ) English No. of seasons 47 ( NFL seasons ) No. of episodes 718 ( games ) Production Executive producer ( s ) Norby Williamson ( 2006 -- present ) Producer ( s ) Roger Lewin Jay Rothman Suzy Kolber ( uncredited ; 2006 -- present ) Location ( s ) Various NFL stadiums ( game telecasts ) ESPN Center , Bristol , Connecticut ( studio segments ) Camera setup Multi-camera Running time 3 - 4 hours Production company ( s ) National Football League ABC Sports ( 1970 -- 2005 ) ESPN ( 2006 -- present ) Release Original network ABC ( 1970 -- 2005 ) ESPN ( 2006 -- present ) Picture format 480i ( SDTV ) ( 1970 -- 2010 ) , 480i ( 16 : 9 SDTV ) ( 2011 -- present ) , 720p ( HDTV ) ( 2011 -- present ) Original release September 21 , 1970 ( 1970 - 09 - 21 ) -- present Chronology Related shows Monday Night Countdown External links Website espn.go.com/nfl/mnf
  Fact 1.0: ESPN Monday Night Football has the genre NFL game telecasts.
  Fact 1.1: ESPN Monday Night Football was created by Roone Arledge.
  Fact 1.2: Chet Forte directed ESPN Monday Night Football from 1970 to 1986.
  Fact 1.3: Larry Kamm directed ESPN Monday Night Football in 1987.
  Fact 1.4: Craig Janoff directed ESPN Monday Night Football from 1988 to 1999.
  Fact 1.5: Drew Esocoff directed ESPN Monday Night Football from 2000 to 2005.
  Fact 1.6: Chip Dean has directed ESPN Monday Night Football from 2006 to the present.
  Fact 1.7: The commentators of ESPN Monday Night Football are Sean McDonough and Lisa Salters.
  Fact 1.8: The studio hosts of ESPN Monday Night Football are Suzy Kolber, Steve Young, Randy Moss, Matt Hasselbeck and Charles Woodson.
  Fact 1.9: The opening theme 'Score' by Bob's Band was used from 1970 to 1975.
  Fact 1.10: The opening theme 'All My Rowdy Friends Are Here on Monday Night' by Hank Williams, Jr. was used from 1989 to 2011 and again from 2017 to the present.
  Fact 1.11: The opening theme 'Heavy Action' by Johnny Pearson was used from 1976 to 1988 and from 2011 to 2016.
  Fact 1.12: 'Heavy Action' was used as the secondary theme from 1989 to 2011 and again from 2017 to the present.
  Fact 1.13: The country of origin of ESPN Monday Night Football is the United States.
  Fact 1.14: The original language of ESPN Monday Night Football is English.
  Fact 1.15: ESPN Monday Night Football has had 47 NFL seasons.
  Fact 1.16: ESPN Monday Night Football has had 718 games as episodes.
  Fact 1.17: Norby Williamson has been executive producer of ESPN Monday Night Football from 2006 to the present.
  Fact 1.18: Roger Lewin and Jay Rothman were producers of ESPN Monday Night Football.
  Fact 1.19: Suzy Kolber is an uncredited producer of ESPN Monday Night Football from 2006 to the present.
  Fact 1.20: ESPN Monday Night Football game telecasts are located at various NFL stadiums.
  Fact 1.21: ESPN Monday Night Football studio segments are produced at ESPN Center in Bristol, Connecticut.
  Fact 1.22: ESPN Monday Night Football uses a multi-camera setup.
  Fact 1.23: ESPN Monday Night Football has a running time of 3 to 4 hours.
  Fact 1.24: The National Football League is a production company of ESPN Monday Night Football.
  Fact 1.25: ABC Sports was a production company of ESPN Monday Night Football from 1970 to 2005.
  Fact 1.26: ESPN was a production company of ESPN Monday Night Football from 2006 to the present.
  Fact 1.27: The original network of ESPN Monday Night Football was ABC from 1970 to 2005.
  Fact 1.28: The original network of ESPN Monday Night Football is ESPN from 2006 to the present.
  Fact 1.29: ESPN Monday Night Football was broadcast in 480i (SDTV) picture format from 1970 to 2010.
  Fact 1.30: ESPN Monday Night Football was broadcast in 480i (16:9 SDTV) picture format from 2011 to the present.
  Fact 1.31: ESPN Monday Night Football was broadcast in 720p (HDTV) picture format from 2011 to the present.
  Fact 1.32: ESPN Monday Night Football originally released on September 21, 1970.
  Fact 1.33: ESPN Monday Night Football has been released from September 21, 1970 to the present.
  Fact 1.34: The related show of ESPN Monday Night Football is Monday Night Countdown.
  Fact 1.35: The website of ESPN Monday Night Football is espn.go.com/nfl/mnf.

Retrieved items:
<item 1>
Monday Night Football
From 2003 to 2005 , one game was played on Thursday and another Monday under the Monday Night Football banner . Starting in 2006 , when the series moved to
cable , two games are played on the opening Monday night to capitalize on fan interest during `` NFL Kickoff Weekend '' . Monday night games early in the season
are often highly anticipated since records are new , teams usually are showcasing fresh talent and potential , and storylines coming into the season are often
played out as fans try to see if these hyped teams are up to form . Since no one knows during the first month of the season if a team is indeed good , or will
rebound from a difficult start , interest is usually high for the first few weeks of the MNF season . Since the MNF schedule is set in April and can not be
changed , the league and network can not guarantee a late season matchup will have any significance or be highly anticipated .
</item 1>

<item 2>
Monday Night Football
wordpress \. com \ b on the global blacklist ESPN Monday Night Football Genre NFL game telecasts Created by Roone Arledge Directed by Chet Forte ( 1970 -- 1986
) Larry Kamm ( 1987 ) Craig Janoff ( 1988 -- 1999 ) Drew Esocoff ( 2000 -- 2005 ) Chip Dean ( 2006 -- present ) Presented by Commentators : Sean McDonough Lisa
Salters Studio hosts : Suzy Kolber Steve Young Randy Moss Matt Hasselbeck Charles Woodson Opening theme `` Score '' by Bob 's Band ( 1970 -- 1975 ) `` All My
Rowdy Friends Are Here on Monday Night '' by Hank Williams , Jr. ( 1989 -- 2011 ; 2017 -- present ) `` Heavy Action '' by Johnny Pearson ( 1976 -- 1988 ; 2011
-- 2016 ) ( used as secondary theme from 1989 to 2011 and again from 2017 -- present ) Country of origin United States Original language ( s ) English No. of
seasons 47 ( NFL seasons ) No.
</item 2>

<item 3>
Monday Night Football
Jr . Songs `` All My Rowdy Friends Are Here on Monday Night '' `` Heavy Action '' `` Start Me Up '' `` The Party 's Over '' Yearly results Monday night NFL
games prior to 1970 1970 -- 1989 1990 -- 2009 2010 -- present National Football League on television and radio Broadcast partners ABC ( AFL ) CBS Fox NBC
Defunct networks DuMont Sports Network Monday Night Football History Monday Night Countdown Results Games prior to 1970 1970 -- 1989 1990 -- 2009 2010 --
present Sunday Night Football ESPN NBC TNT Results ESPN NBC TNT Pregame TV programs The NFL Today CBS Fox NFL Sunday ESPN Sunday NFL Countdown Monday Night
Countdown NBC The NFL on NBC pregame show Football Night in America NFL Network List of programs NFL AM NFL Classics NFL GameDay NFL Replay NFL Total Access The
Timeline Thursday Night Football Results ( 2006 -- present ) 2007 New England Patriots -- New York Giants game Miracle in Motown NFL Films TV programs Football
Follies A Football Life Full Color Football Hard Knocks Inside the NFL NFL Dream Season NFL Films Game of the Week NFL 's Greatest Games NFL Matchup NFL Top 10
NFL Top 100 Other
</item 3>

<item 4>
Monday Night Football
Monday Night Football - Wikipedia Monday Night Football Jump to : navigation , search For other uses , see Monday Night Football ( disambiguation ) . `` NFL on
ESPN '' redirects here . For the program aired between 1987 and 2005 , see ESPN Sunday Night Football . An automated process has detected links on this page on
the local or global blacklist . If the links are appropriate you may request whitelisting by following these instructions ; otherwise consider removing or
replacing them with more appropriate links . ( To hide this tag , set the `` invisible '' field to `` true '' ) List of blacklisted links : ( show )
http://nfllabor.files.wordpress.com/2013/12/week-161.pdf Triggered by \ bfiles \. wordpress \. com \ b on the global blacklist
https://web.archive.org/web/20131224083130/http://nfllabor.files.wordpress.com/2013/12/week-161.pdf Triggered by \ bfiles \. wordpress \.
</item 4>

<item 5>
Monday Night Football
However , the forecasting abilities of the NFL 's scheduling officials have exhibited uncanny proficiency . Many MNF games occurring later in the season feature
at least one team that is either headed to the playoffs or must win the MNF game to clinch a playoff spot . The problem of having a national spotlight game
which during the season 's most critical weeks late in the year probably would not show the most important game of the week was long known by the league and
network . As a result of this , the NFL wished to move the `` Game of the Week '' idea to Sunday nights as to make flex scheduling possible . This was a move
which would ultimately mean the end of Monday Night Football on ABC ( cable games are protected from the NFL 's flexible scheduling rule adopted for the 2006 --
07 season ; the new rule applies only to CBS , Fox , and NBC 's Sunday night games ) . The league currently has no interest in expanding flexible scheduling to
include Monday Night Football , citing the logistical issues of moving games back and forth between Sundays and Mondays . Franchises with the most appearances (
edit ) The franchise with the most Monday night appearances is the Miami Dolphins .
</item 5>

<item 6>
Monday Night Football
go.com/nfl/mnf ESPN Monday Night Football ( abbreviated as MNF and also known as ESPN Monday Night Football on ABC for rare live special broadcast ) is a live
television broadcast of weekly National Football League ( NFL ) games on ESPN in the United States . From 1970 to 2005 , it aired on sister broadcast network
ABC . Monday Night Football was , along with Hallmark Hall of Fame and the Walt Disney anthology television series , one of the longest - running prime time
programs ever on commercial network television , and one of the highest - rated , particularly among male viewers . MNF is preceded on air by Monday Night
Countdown served by Applebee 's . Monday Night Football is also broadcast in Canada on TSN and RDS , and in most of Europe . On September 7 , 2013 , the NFL
announced that British Eurosport would show Monday Night Football games live in the United Kingdom for the 2013 and 2014 seasons . The telecasts are also seen
in most of Australia on ESPN Australia , in Portugal on SportTV 3 and SportTV HD and on TV 2 Sport in Denmark , and in some other regions of the world outside
the U.S.
</item 6>

<item 7>
Monday Night Football
The layout is standard , with teams on the background of their color , with timeout indicators also displaying in team colors . Experimentally , the score bar
`` auto - hides '' itself during plays , though it is shown permanently during the final two minutes of each half and , if necessary , overtime . This is also
the first time that ESPN has used the full 16 : 9 frame for its NFL graphics . Scoring records ( edit ) See also : List of Monday Night Football results ( 1970
-- 89 ) , List of Monday Night Football results ( 1990 -- 2009 ) , and List of Monday Night Football results ( 2010 -- present ) Most points 59 -- Philadelphia
Eagles , November 15 , 2010 vs. Washington Redskins 55 -- Indianapolis Colts , October 31 , 1988 vs. Denver Broncos 52 -- San Francisco 49ers , December 23 ,
1991 vs. Chicago Bears 52 -- San Diego Chargers , December 20 , 1982 vs. Cincinnati Bengals 51 -- New Orleans Saints , November 24 , 2008 vs.
</item 7>

<item 8>
Monday Night Football
As of 2008 , Westwood One was no longer controlled by CBS , but the network retained its NFL broadcast rights . In 2011 , Westwood One was purchased by Dial
Global . Then in 2014 , Dial Global , including Westwood One , was acquired by Cumulus Media . The Spanish - language broadcast is carried on ESPN Deportes
Radio . Nielsen ratings ( edit ) This article needs to be updated . Please update this article to reflect recent events or newly available information . (
December 2016 ) The highest - rated Monday Night Football telecast on ABC was the Miami Dolphins ' victory over the previously undefeated Chicago Bears on
December 2 , 1985 , which drew a national Nielsen rating of 29.6 and a share of 46 . ABC 's lowest - rated MNF game was the St. Louis Rams ' defeat of the Tampa
Bay Buccaneers on October 18 , 2004 , which drew a 7.7 rating . During the 1995 -- 1996 television season , Monday Night Football averaged a 17.1 household
rating . Its main competitor , Murphy Brown on CBS , averaged a 12.3 rating .
</item 8>

<item 9>
Monday Night Football
6 2016 summary 2.2. 7 2017 summary 3 Theme music 4 Digital on - screen graphics 4.1 1997 -- 2005 4.2 2006 -- present 5 Scoring records 6 Scheduling on local
stations 7 Commentators 8 Foreign - language versions 8.1 Spanish version 8.2 Portuguese version 9 Radio broadcasts 10 Nielsen ratings 11 Additional NFL game
rights 11.1 Playoff games 11.2 Pro Bowl 12 See also 13 Footnotes 14 References 15 Further reading 16 External links Overview ( edit ) This section needs
expansion . You can help by adding to it . ( November 2010 ) See also : History of Monday Night Football After 45 seasons , there now have been a total of 730
games televised by the Monday Night Football franchise . Scheduling problems ( edit ) To avoid any scheduling unfairness where , just before the first playoff
game , a team may have five days off and others six , there is no Monday night game during the final week of the regular season .
</item 9>

<item 10>
Monday Night Football
1987 -- 2005 ) NFL Draft NFL on ABC Sports Football Sunday on ESPN Radio Studio programs The Blitz Monday Night Countdown NFL Matchup NFL Primetime Sunday NFL
Countdown NFL Live NFL Insiders Other football programs Arena Football League on ESPN CFL on TSN ESPN College Football USFL on ESPN Related articles History of
Monday Night Football NFL on television ( history ) Commentators Sean McDonough Play - by - play TBA Color commentary Lisa Salters Sideline reporter Former
commentators Bonnie Bernstein Mike Ditka Jon Gruden Ron Jaworski Suzy Kolber Tony Kornheiser Michele Tafoya Joe Theismann Mike Tirico Dick Vermeil Steve Young
Lore televised by ESPN ESPN takes over MNF 2005 Saints -- Giants / Cowboys -- Redskins Doubleheader 2006 Bears -- Cardinals Comeback Fail Mary Sunday Night
Football era American Bowl ( 1990 - 2005 ) Fútbol Americano Pro Bowls broadcast on ESPN 1988 1990 1991 1992 1994 2005 2006 2015 2016 2017 2018 Music Edd
Kalehoff Johnny Pearson Hank Williams , Jr . Songs `` All My Rowdy Friends Are Here on Monday Night '' `` Heavy
</item 10>
=== END ITEM id=f17baa273fed3850 ===

=== ITEM id=69fc40e48cbab7cc ===
Gold sentences and facts:
Sentence 0: In August 2015 , the series was reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop other series with Marvel instead .
  Fact 0.0: In August 2015, the series was reinvented as Most Wanted.
  Fact 0.1: The series Most Wanted received a pilot order from ABC in August 2015.
  Fact 0.2: The cast was filled out in early 2016.
  Fact 0.3: Production was completed by late March.
  Fact 0.4: The episode was co-produced by ABC Studios and Marvel Television.
  Fact 0.5: In May 2016, ABC announced that it would not pick up the pilot to series.
  Fact 0.6: In May 2016, ABC announced that it would look to develop other series with Marvel instead.

Retrieved items:
<item 1>
Most Wanted (TV pilot)
^ Jump up to : `` Marvel 's Most Wanted '' . MckennaGrace.com . May 3 , 2016 . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017 . ^ Jump
up to : Frederick , Brittany . `` Who directed Spider - Man : Homecoming ? '' . Fansided . Archived from the original on July 14 , 2017 . Retrieved July 14 ,
2017 . ^ Jump up to : Walsh , S.M. ( June 6 , 2016 ) . `` ' Stitchers ' Season 3 : Meet the Cast '' . Heavy.com . Archived from the original on July 14 , 2017 .
Retrieved July 14 , 2017 . Jump up ^ Daniels , Hunter ( July 23 , 2011 ) . `` Comic - Con : Marvel TV Announces Development Slate ; Includes Live - Action AKA
Jessica Jones ( Alias ) , The Hulk , Cloak And Dagger , Mockingbird , Animated Hulk And The Agents Of S.M.A.S.H. , More '' .
</item 1>

<item 2>
Most Wanted (TV pilot)
The series would have been based on storylines occurring at the end of that second season , but ABC passed on the project . In August 2015 , the series was
reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode
was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop
other series with Marvel instead . Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Main 2.2 Guests 3 Production 3.1 Development 3.2 Writing 3.3 Casting
3.4 Filming 3.5 Marvel Cinematic Universe tie - ins 4 No pick - up and future 5 References 6 External links Premise ( edit ) Ex-spies and ex-spouses Bobbi Morse
and Lance Hunter are on the run trying to uncover a conspiracy against them . With no help from S.H.I.E.L.D.
</item 2>

<item 3>
Most Wanted (TV pilot)
^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) . `` How Will ' Agents of SHIELD ' Set Up Spinoff ' Marvel 's Most Wanted ' ? '' . The Hollywood Reporter
. Archived from the original on August 20 , 2016 . Retrieved August 20 , 2016 . Jump up ^ Carinacas , Peter ( February 17 , 2016 ) . `` Artisans Booked on '
Marvel 's Most Wanted , ' ' Pearl , ' ' Dirty Dancing ' and ' Stronger ' '' . Variety . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017
. Jump up ^ Carinacas , Peter ( March 15 , 2016 ) . `` Agencies Book Artisans on ' Thor : Ragnarok , ' ' Sugar Queen , ' ' Chagrin Falls ' and ' Bunker Hill '
'' . Variety . Archived from the original on March 20 , 2016 . Retrieved March 20 , 2016 .
</item 3>

<item 4>
Most Wanted (TV pilot)
Retrieved March 25 , 2016 . ^ Jump up to : Abrams , Natalie ( July 25 , 2014 ) . `` Agents of S.H.I.E.L.D. Reveals New Characters , Carter Adds Captain
Directors '' . TV Guide . Archived from the original on July 26 , 2014 . Retrieved July 25 , 2014 . ^ Jump up to : Andreeva , Nellie ( January 26 , 2016 ) . ``
' Marvel 's Most Wanted ' ABC Pilot Casts Delroy Lindo As Dominic Fortune '' . Deadline.com . Archived from the original on January 26 , 2016 . Retrieved
January 27 , 2016 . Jump up ^ Huver , Scott ( April 22 , 2016 ) . `` '' Marvel 's Most Wanted '' Co-Star Andrade Promises An `` Unpredictable '' Series `` .
Comic Book Resources . Archived from the original on April 22 , 2016 . Retrieved April 22 , 2016 . ^ Jump up to : `` Marvel 's Most Wanted '' .
</item 4>

<item 5>
Most Wanted (TV pilot)
At that time , the producers were prepping `` if - come '' orders , `` deals with writers that will come into effect in the event a pilot is picked ( up ) to
series '' . On May 12 , 2016 , the pilot was passed on by ABC , with Lee 's successor Channing Dungey explaining that Most Wanted `` did not feel as strong as
some of the other pilots that we shot . We talked about it with Marvel and we all came to an agreement that we want to figure out what the next show is that we
do together , is something that we all feel is as creatively strong as it can be . '' However , Dungey said that ABC was open to the series being shopped to
another network if Marvel found `` another home '' for it . Loeb also acknowledged this possibility , but said that the series was designed `` to do a very
specific thing '' alongside S.H.I.E.L.D. on ABC . References ( edit ) ^ Jump up to : `` 2016 ABC Pilots '' . Deadline.com . February 22 , 2016 . Archived from
the original on February 23 , 2016 .
</item 5>

<item 6>
Most Wanted (TV pilot)
^ Jump up to : Schwartz , Terri ( March 22 , 2016 ) . `` Agents of SHIELD EP Jeff Bell on Mockingbird and Hunter Departure and Marvel 's Most Wanted '' . IGN .
Archived from the original on March 23 , 2016 . Retrieved March 23 , 2016 . Jump up ^ Schwartz , Terri ( January 9 , 2016 ) . `` Agents of SHIELD Showrunners
Explain Marvel 's Most Wanted Spinoff Plan '' . IGN . Archived from the original on January 10 , 2016 . Retrieved January 10 , 2016 . ^ Jump up to : Lovett ,
Jamie ( October 27 , 2015 ) . `` EXCLUSIVE Agents Of S.H.I.E.L.D. EP Talks Secret Warriors , Civil War , Last Week 's Shocking Death & More '' . ComicBook.com .
Archived from the original on October 27 , 2015 . Retrieved October 28 , 2015 . ^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) .
</item 6>

<item 7>
Most Wanted (TV pilot)
Retrieved August 9 , 2016 . External links ( edit ) Marvel 's Most Wanted on IMDb Marvel Cinematic Universe Feature films Phase One Iron Man The Incredible Hulk
Iron Man 2 Thor Captain America : The First Avenger The Avengers accolades Phase Two Iron Man 3 Thor : The Dark World Captain America : The Winter Soldier
Guardians of the Galaxy Avengers : Age of Ultron Ant - Man Phase Three Captain America : Civil War Doctor Strange Guardians of the Galaxy Vol. 2 Spider - Man :
Homecoming Thor : Ragnarok Black Panther Avengers : Infinity War production Ant - Man and the Wasp Untitled Avengers film production Television series ABC
Agents of S.H.I.E.L.D. episodes season 1 5 characters Agent Carter season 1 characters Inhumans `` Behold ... The Inhumans '' `` Those Who Would Destroy Us ''
Netflix Daredevil season 1 characters Jessica Jones season 1 characters Luke Cage season 1 characters Iron Fist season 1 characters The Defenders The Punisher
Hulu Runaways Freeform Cloak & Dagger Unaired pilot Most Wanted Other Short films Item 47 Agent Carter All Hail the King Digital series WHIH Newsfront Agents of
S.H.I.E.L.D.
</item 7>

<item 8>
Most Wanted (TV pilot)
In August 2015 , the spin - off series received new life , reworked as Marvel 's Most Wanted and given a pilot order . Bell and Zbyszewski were again developing
the series , writing the pilot and executive producing . The pair were set to serve as showrunners for the series , if it was picked up . Loeb and Jim Chory
were also attached as executive producers . The series would still focus on Morse and Hunter , with Palicki and Blood both attached , and was described as `` a
new take focusing on the same duo and their continuing adventures . '' In January 2016 , Lee stated that the pilot was `` absolutely '' moving forward ,
praising the script and saying production would begin `` in the next few months . '' In the Agents of S.H.I.E.L.D. episode `` Parting Shot '' , the two
characters were written off that series , having them leave and disavow S.H.I.E.L.D.
</item 8>

<item 9>
Most Wanted (TV pilot)
Archived from the original on February 23 , 2016 . Retrieved February 22 , 2016 . ^ Jump up to : `` Oded Fehr and Fernanda Andrade Join ' Marvel 's Most Wanted
' '' . Marvel.com . March 25 , 2016 . Archived from the original on March 26 , 2016 . Retrieved March 25 , 2016 . ^ Jump up to : Khatchatourian , Maane (
February 14 , 2015 ) . `` ' Agents of SHIELD ' Promotes Adrianne Palicki to Series Regular '' . Variety . Archived from the original on February 15 , 2015 .
Retrieved February 15 , 2015 . ^ Jump up to : Abrams , Natalie ( March 23 , 2016 ) . `` Agents of S.H.I.E.L.D. 's Adrianne Palicki , Nick Blood on that
emotional exit '' . Entertainment Weekly . Archived from the original on March 25 , 2016 . Retrieved March 25 , 2016 .
</item 9>

<item 10>
Most Wanted (TV pilot)
Most Wanted ( TV pilot ) - wikipedia Most Wanted ( TV pilot ) Jump to : navigation , search Most Wanted Genre Action / Adventure Science fiction Superhero Drama
Created by Jeffrey Bell Paul Zbyszewski Based on Mockingbird by Gerry Conway Barry Smith Lance Hunter by Gary Friedrich Herb Trimpe Starring Adrianne Palicki
Nick Blood Delroy Lindo Fernanda Andrade Oded Fehr Country of origin United States Original language ( s ) English Production Executive producer ( s ) Jeffrey
Bell Paul Zbyszewski Jeph Loeb Jim Chory Production company ( s ) ABC Studios Marvel Television Distributor Disney -- ABC Domestic Television Release Original
network ABC Picture format 720p ( HDTV ) Audio format 5.1 surround sound Chronology Related shows Marvel 's Agents of S.H.I.E.L.D. Marvel Cinematic Universe
television series Marvel 's Most Wanted , or simply Most Wanted , is an unaired American television pilot based on the Marvel Comics characters Bobbi Morse and
Lance Hunter .
</item 10>
=== END ITEM id=69fc40e48cbab7cc ===

=== ITEM id=ed26c8950e4b7207 ===
Gold sentences and facts:
Sentence 0: The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of law by aggrieved citizens .
  Fact 0.0: The provisions relating to the right to property were changed a number of times.
  Fact 0.1: The Forty-Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights.
  Fact 0.2: A new provision, Article 300-A, was added to the constitution.
  Fact 0.3: Article 300-A provides that no person shall be deprived of his property save by authority of law.
  Fact 0.4: If a legislature makes a law depriving a person of his property, the State has no obligation to pay anything as compensation.
  Fact 0.5: An aggrieved person shall have no right to move the court under Article 32 in that case.
  Fact 0.6: The right to property is no longer a fundamental right.
  Fact 0.7: The right to property is still a constitutional right.
  Fact 0.8: If the government appears to have acted unfairly, the action can be challenged in a court of law by aggrieved citizens.

Retrieved items:
<item 1>
Fundamental rights in India
Jump up ^ Sovereign means possessing supreme or ultimate power . `` in modern democracies the people 's will is in theory sovereign '' synonyms : supreme ,
absolute , unlimited , unrestricted , unrestrained , unbounded , boundless , infinite , ultimate , total , unconditional , full , utter , paramount ; principal
, chief , dominant , predominant Jump up ^ `` What causes judicial delay ? Judgements diluting time frames in Code of Civil Procedure worsen the problem of
adjournments '' . Retrieved 13 September 2017 . Jump up ^ `` Paras 316 and 317 of Kesavananda Bharati v. State of Kerala , ( AIR 1973 SC 1461 ) '' . 1973 .
Retrieved 21 October 2016 . Jump up ^ `` IX Schedule laws open to review '' . 2007 . Retrieved 21 October 2017 . Jump up ^ 44th Amendment Act , 1978 . Jump up ^
Mahapatra , Dhananjay ( 28 February 2009 ) . `` Should right to property return ? '' . The Times of India . Retrieved 8 July 2010 .
</item 1>

<item 2>
Fundamental rights in India
The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the
list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his
property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the
State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer
a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of
law by aggrieved citizens . The liberalisation of the economy and the government 's initiative to set up special economic zones has led to many protests by
farmers and have led to calls for the reinstatement of the fundamental right to private property .
</item 2>

<item 3>
Fundamental rights in India
It is the principal foundation of all other rights and liberties , and guarantees the following : Equality before law : Article 14 of the constitution
guarantees that all people shall be equally protected by the laws of the country . It means that the State will treat people in the same circumstances alike .
This article also means that individuals , whether citizens of India or otherwise shall be treated differently if the circumstances are different . Social
equality and equal access to public areas : Article 15 of the constitution states that no person shall be discriminated on the basis of religion , race , caste
, sex or place of birth . Every person shall have equal access to public places like public parks , museums , wells , bathing ghats and temples etc . However ,
the State may make any special provision for women and children . Special provisions may be made for the advancements of any socially or educationally backward
class or scheduled castes or scheduled tribes . Equality in matters of public employment : Article 16 of the constitution lays down that the State can not
discriminate against anyone in the matters of employment . All citizens can apply for government jobs . There are some exceptions .
</item 3>

<item 4>
Fundamental rights in India
In 2007 , Supreme Court ruled that there could not be any blanket immunity from judicial review for the laws inserted in the Ninth Schedule . Apex court also
stated it shall examine laws included in the Ninth Schedule after 1973 for any incompatibility with the basic structure doctrine . Amendment to Article 31C (
edit ) Section 4 of the 42nd Amendment , had changed Article 31C of the constitution to accord precedence to the Directive Principles ( earlier applicable only
to clauses b & c of Article 39 ) over the fundamental rights of individuals . In Minerva Mills v. Union of India case , supreme court ruled that the amendment
to the Article 31C is not valid and ultra vires . Right to property ( edit ) The Constitution originally provided for the right to property under Articles 19
and 31 . Article 19 guaranteed to all citizens the right to acquire , hold and dispose of property . Article 31 provided that `` no person shall be deprived of
his property save by authority of law . '' It also provided that compensation would be paid to a person whose property has been taken for public purposes .
</item 4>

<item 5>
Fundamental rights in India
The six fundamental rights recognised by the Indian constitution are the right to equality , right to freedom , right against exploitation , right to freedom of
religion , cultural and educational rights , right to constitutional remedies and right to privacy . The right to equality includes equality before law ,
prohibition of discrimination on grounds of religion , race , caste , gender or place of birth , and equality of opportunity in matters of employment ,
abolition of untouchability and abolition of titles . The right to freedom includes freedom of speech and expression , assembly , association or union or
cooperatives , movement , residence , and right to practice any profession or occupation , right to life and liberty , protection in respect to conviction in
offences and protection against arrest and detention in certain cases . The right against exploitation prohibits all forms of forced labour , child labour and
trafficking of human beings . The right to freedom of religion includes freedom of conscience and free profession , practice , and propagation of religion ,
freedom to manage religious affairs , freedom from certain taxes and freedom from religious instructions in certain educational institutes . Cultural and
educational rights preserve the right of any section of citizens to conserve their culture , language or script , and right of minorities to establish and
administer educational institutions of their choice .
</item 5>

<item 6>
Fundamental rights in India
. The Times of India . Retrieved 8 July 2010 . Jump up ^ `` Court rejects plea to make property a fundamental right '' . The Hindu . Chennai , India . 19
October 2010 . Jump up ^ `` Education is now a fundamental right of every child '' . Hindustan Times . 1 April 2010 . Archived from the original on 9 July 2010
. Retrieved 8 July 2010 . Footnotes ( edit ) Links to related articles ( hide ) Government of India Constitution Constitution Fundamental Rights , Directive
Principles and Fundamental Duties Federalism ( Union List State List Concurrent List ) Executive Union Government Prime Minister ( list ) Cabinet Ministers
President ( list ) Vice President ( list ) Civil Services All India Services ( IAS IFS / IFoS IPS ) Cabinet Secretary Secretaries ( Defence Finance Foreign Home
) State governments Chief Minister ( list ) Chief Secretary Governors ( list ) Legislative Parliament Lok Sabha ( Speaker ) Rajya Sabha ( Chairman ) Vidhan
Sabha Vidhan Parishad Leader of the Opposition Judiciary Supreme Court ( Chief Justice ) High Courts District Courts Retrieved from ``
https://en.wikipedia.org/w/index.
</item 6>

<item 7>
Fundamental rights in India
They are covered in Part III ( Articles 12 to 35 ) of Indian constitution . Contents ( hide ) 1 Genesis 2 Significance and characteristics 3 Right to equality 4
Right to freedom 5 Right against exploitation 6 Right to freedom of religion 7 Right to life 8 Cultural and Educational rights 9 Right to constitutional
remedies 10 Right to Privacy 11 Critical analysis 12 Amendments 13 Fundamental rights not sacrosanct 13.1 Validity of Article 31B 13.2 Amendment to Article 31C
13.3 Right to property 13.4 Right to education 14 See also 15 References 16 Footnotes Genesis ( edit ) The development of constitutionally guaranteed
fundamental human rights in India was inspired by historical examples such as England 's Bill of Rights ( 1689 ) , the United States Bill of Rights ( approved
on 17 September 1787 , final ratification on 15 December 1791 ) and France 's Declaration of the Rights of Man ( created during the revolution of 1789 , and
ratified on 26 August 1789 ) .
</item 7>

<item 8>
Fundamental rights in India
The right to constitutional remedies is present for enforcement of Fundamental Rights . The right to privacy is an intrinsic part of Article 21 that protects
life and liberty of the citizens . Fundamental rights for Indians have also been aimed at overturning the inequalities of pre-independence social practices .
Specifically , they have also been used to abolish untouchability and thus prohibit discrimination on the grounds of religion , race , caste , sex , or place of
birth . They also forbid trafficking of human beings and forced labour ( a crime ) . They also protect cultural and educational rights of religious and
linguistic minorities by allowing them to preserve their languages and also establish and administer their own education institutions . They are covered in Part
III ( Articles 12 to 35 ) of Indian constitution .
</item 8>

<item 9>
Fundamental rights in India
These fundamental rights help not only in protection but also the prevention of gross violations of human rights . They emphasise on the fundamental unity of
India by guaranteeing to all citizens the access and use of the same facilities , irrespective of background . Some fundamental rights apply for persons of any
nationality whereas others are available only to the citizens of India . The right to life and personal liberty is available to all people and so is the right
to freedom of religion . On the other hand , freedoms of speech and expression and freedom to reside and settle in any part of the country are reserved to
citizens alone , including non-resident Indian citizens . The right to equality in matters of public employment can not be conferred to overseas citizens of
India . Fundamental rights primarily protect individuals from any arbitrary state actions , but some rights are enforceable against individuals . For instance ,
the Constitution abolishes untouchability and also prohibits begar . These provisions act as a check both on state action as well as the action of private
individuals . However , these rights are not absolute or uncontrolled and are subject to reasonable restrictions as necessary for the protection of general
welfare . They can also be selectively curtailed .
</item 9>

<item 10>
Fundamental rights in India
72 ∙ 73 ∙ 74 ∙ 75 ∙ 76 ∙ 77 ∙ 78 ∙ 79 ∙ 80 ∙ 81 ∙ 82 ∙ 83 ∙ 84 ∙ 85 ∙ 86 ∙ 87 ∙ 88 ∙ 89 ∙ 90 ∙ 91 ∙ 92 ∙ 93 ∙ 94 ∙ 95 ∙ 96 ∙ 97 ∙ 98 ∙ 99 ∙ 100 ∙ 101 Related
topics ( show ) Union List State List Concurrent List Basic structure doctrine Uniform civil code Fundamental Rights are the basic rights of the people and the
charter of rights contained in Part III ( Article 12 to 35 ) of Constitution of India . It guarantees civil liberties such that all Indians can lead their lives
in peace and harmony as citizens of India .
</item 10>
=== END ITEM id=ed26c8950e4b7207 ===

=== ITEM id=b4435db1e452342a ===
Gold sentences and facts:
Sentence 0: Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double Down
  Fact 0.0: The title is Diary of a Wimpy Kid: The Getaway.
  Fact 0.1: The author is Jeff Kinney.
  Fact 0.2: The country is United States.
  Fact 0.3: The language is English.
  Fact 0.4: The series is Diary of a Wimpy Kid.
  Fact 0.5: The genres are children's novel and graphic novel.
  Fact 0.6: The publisher is Amulet Books (US).
  Fact 0.7: The publisher is Puffin Books (UK).
  Fact 0.8: The publication date is November 7, 2017 (worldwide).
  Fact 0.9: The media type is print (hardcover).
  Fact 0.10: The book has 217 story pages, 224 altogether.
  Fact 0.11: The ISBN is 978-1-4197-2545-6.
  Fact 0.12: It is preceded by Double Down.

Retrieved items:
<item 1>
Diary of a Wimpy Kid: The Getaway
They each wear some of the clothes in there ; they first head to the clubhouse , and then the putting green for golf . However , they are spotted by a security
guard , who escorts them out of the green . A family had reported them for wearing their missing clothes . They attempted a failed getaway with a golf cart ,
and are ultimately sent back to the hotel to return the clothes . They were also ejected from the hotel for theft , so they were forced to stay overnight at the
airport hotel . The next day , Susan refuses to leave the resort to leave on a high note . So they sneak in . They skip the pool and then find Rodrick 's
girlfriend with a man named Rodrigo . They go back to the beach , and scavenge for food by distracting others and then eating their food . They go to take one
more photo , but are spotted by the family who they took clothes from . The Heffleys try to escape the resort . Greg heads for the beach and windsurfs away . He
gets stuck in the dolphin swimming area , but is chased back to shore by security .
</item 1>

<item 2>
Diary of a Wimpy Kid: The Getaway
Diary of a Wimpy Kid : the Getaway - wikipedia Diary of a Wimpy Kid : the Getaway Jump to : navigation , search ( hide ) This article has multiple issues .
Please help improve it or discuss these issues on the talk page . ( Learn how and when to remove these template messages ) This article consists almost entirely
of a plot summary . It should be expanded to provide more balanced coverage that includes real - world context . Please edit the article to focus on discussing
the work rather than merely reiterating the plot . ( August 2016 ) ( Learn how and when to remove this template message ) This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
December 2017 ) ( Learn how and when to remove this template message ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page .
</item 2>

<item 3>
Diary of a Wimpy Kid: The Getaway
The book was unveiled during the 2017 Diary of a Wimpy Kid Virtually Live Event which was live streamed via YouTube as part of the 10th anniversary of the first
book . The book was published and released on November 7 , 2017 . Contents ( hide ) 1 Plot 2 Release date 3 Reception 4 References Plot ( edit ) After many
frustrations , Greg Heffley 's parents decide to skip Christmas and travel by plane to Isla De Corales , the hotel they stayed at while on their honeymoon much
to Greg 's dismay . On Christmas Eve they drive to the airport , but are delayed by a series of mishaps . They encounter holiday traffic but arrive at the
airport . As they fail to unload in time at the drop off zone , Frank and Greg have to return later with the luggage . Greg steps in a slush puddle and they end
up missing the shuttle . In the airport , the family 's suitcases are too heavy , so they stuff some clothes into their carry on baggage .
</item 3>

<item 4>
Diary of a Wimpy Kid: The Getaway
He is even excited by the activities the resort offer which he sees on the bus . However , Frank and Susan are disappointed that the resort has since changed ,
and it is now split with a family side ( the Mild Side ) and a couple side ( the Wild Side ) . The family also discovers that they got the wrong luggage that is
locked , and their luggage was sent back home . They only have the clothes they put in their carry on luggage . They attempt to buy some new clothes , but Frank
wo n't buy them as they are too expensive . The family goes to explore the resort and end up at the pool . The Director of Fun shows up , however , and gets
everybody moving . Greg is forced to take part in two games . After this , Greg sees an iguana by the pool and is worried , so the family goes to get some lunch
. Greg and his family try to eat outside ; however , too many birds and bugs keep ruining their meals . Frank contracts a stomach bug from his water , so they
head back to the suite . The next day , Greg uses the shower which is outside .
</item 4>

<item 5>
Diary of a Wimpy Kid: The Getaway
Please discuss this issue on the article 's talk page . ( December 2017 ) This article needs more links to other articles to help integrate it into the
encyclopedia . Please help improve this article by adding links that are relevant to the context within the existing text . ( December 2017 ) ( Learn how and
when to remove this template message ) ( Learn how and when to remove this template message ) Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country
United States Language English Series Diary of a Wimpy Kid Genre Children 's novel Graphic Novel Publisher Amulet Books ( US ) Puffin Books ( UK ) Publication
date November 7 , 2017 ( Worldwide ) Media type Print ( hardcover ) Pages 217 story pages ( 224 altogether ) ISBN 978 - 1 - 4197 - 2545 - 6 Preceded by Double
Down Diary of a Wimpy Kid : The Getaway is the twelfth book in the Diary of a Wimpy Kid series by Jeff Kinney .
</item 5>

<item 6>
Diary of a Wimpy Kid: The Getaway
The couple sit on either side of him with their baby . Greg becomes paranoid when he watches the plane 's safety video and begins to ask questions about
unfortunate events . Greg is very aggravated of the other passengers because most are n't watching the safety video . Mid flight , he feels boxed in when the
people in front recline and the person behind slides his feet on Greg 's armrest . Greg tries to recline his chair too but when he asks the flight attendant how
, he learns that he ca n't because the row behind him is an emergency exit . He tries to swap seats with Frank but Rodrick beats him to it . He stays in the
bathroom because the couple 's baby started to get annoying , but then he returns to his seat after experiencing turbulence . He somehow was able to sleep for
the rest of the flight . As they land on Christmas Day , Greg begins to understand why the family came there . They touch down and began to head to the resort ,
Isla de Corales . He is even excited by the activities the resort offer which he sees on the bus .
</item 6>

<item 7>
Diary of a Wimpy Kid: The Getaway
Waterstones.com . Retrieved 10 December 2017 . ( hide ) The Diary of a Wimpy Kid series by Jeff Kinney Novels Diary of a Wimpy Kid Rodrick Rules The Last Straw
Dog Days The Ugly Truth Cabin Fever The Third Wheel Hard Luck The Long Haul Old School Double Down The Getaway Characters Greg Heffley Film series Diary of a
Wimpy Kid Rodrick Rules Dog Days The Long Haul Retrieved from `` https://en.wikipedia.org/w/index.
</item 7>

<item 8>
Diary of a Wimpy Kid: The Getaway
They first try the banana boat , but this is ruined when a kid using a water trampoline lands on and bursts their float . They get hungry , so they go to the
golf clubhouse which turns them down as they are not following the dress code . They go to the swim - up bar , and learn of a monkey which has been returning
there for drinks every day . They are chased away when Manny finds the TV remote , changes the channel to his favorite cartoons , and throws the remote in the
water . Greg goes to the arcade , and the family regroup later to find Rodrick . The same girl he helped earlier is now his girlfriend . In the evening , they
go to the beach bonfire and go to see turtles hatching . Just as they are prepared to leave , they begin to hatch , but they lose their way to the sea as people
are using the flashes on their phones to take photos . The next day , Susan goes to the spa and Frank goes to the gym . Rodrick then goes back to the teen zone
.
</item 8>

<item 9>
Diary of a Wimpy Kid: The Getaway
They enter the line for security , but a child unclips a barrier separating the lines , and everybody gets scrambled . They go through security and are delayed
again by somebody who has to remove all of his piercings . Manny then gets his toy stuck in an escalator , delaying them more . As a result , the family panic ,
grab some luggage and dash to the gate . It turns out their flight is delayed . Greg goes to dry his sock but destroys it while doing so . He hears that the
flight is overbooked , and tries to stop his family boarding in exchange of a hotel stay and $300 ( soon $500 ) , but is stopped by Susan . A replacement crew
for the plane is required and they finally board . Greg is first impressed by the plane 's first class seats before he discovers that he has to sit in economy .
However , Frank collected enough air miles because of his business trips to sit in first class . In economy , Mom , Rodrick , and Manny are seated in the right
row . Greg finds out he has three seats to himself , until a couple with a baby arrives late on board .
</item 9>

<item 10>
Diary of a Wimpy Kid: The Getaway
Release date ( edit ) The Getaway is the 12th book in the series , and was released everywhere except Canada and China on November 7 , 2017 . Reception ( edit )
A week since its release , it has received 4.5 / 5 stars on Waterstones with 786 reviews . References ( edit ) Jump up ^ Wimpy Kid ( 27 April 2017 ) . `` Wimpy
Kid LIVE Special 10th Anniversary Presentation and Cover Reveal of Book 12 ! '' . YouTube . Retrieved 10 December 2017 . Jump up ^ `` Cover Of New Diary Of A
Wimpy Kid Book Revealed By Jeff Kinney To Fans Worldwide At Live streamed Carnegie Hall Event Hosted By ABRAMS '' . Tristateupdate.com . Retrieved 10 December
2017 . Jump up ^ `` Diary of a Wimpy Kid : The Getaway by Jeff Kinney -- Waterstones '' . Waterstones.com . Retrieved 10 December 2017 .
</item 10>
=== END ITEM id=b4435db1e452342a ===
