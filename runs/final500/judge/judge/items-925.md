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

This file has 7 items.

=== ITEM id=8e802c1fc02664b8 ===
Gold sentences and facts:
Sentence 0: The 2017 -- 18 NBA season is the 72nd season of the National Basketball Association ( NBA ) . The regular season began on October 17 , 2017 , earlier than previous seasons to reduce the number of `` back - to - back '' games teams are scheduled to play , with the 2017 Eastern Conference champion ( and Finals runner -- up ) Cleveland Cavaliers hosting a game against the Boston Celtics at Quicken Loans Arena in Cleveland , Ohio Christmas games were played on December 25 , 2017 . The 2018 NBA All - Star Game was played on February 18 , 2018 , at the Staples Center in Los Angeles , California . LeBron James of the Cleveland Cavaliers was named the All - Star Game Most Valuable Player . The regular season ended on April 11 , 2018 and the playoffs began on April 14 , 2018 .
  Fact 0.0: The 2017-18 NBA season is the 72nd season of the National Basketball Association (NBA).
  Fact 0.1: The 2017-18 NBA regular season began on October 17, 2017.
  Fact 0.2: The 2017-18 NBA regular season began earlier than previous seasons.
  Fact 0.3: The earlier start of the 2017-18 season was meant to reduce the number of 'back-to-back' games teams are scheduled to play.
  Fact 0.4: The 2017 Eastern Conference champion Cleveland Cavaliers, who were also Finals runner-up, hosted a game against the Boston Celtics at Quicken Loans Arena in Cleveland, Ohio.
  Fact 0.5: Christmas games were played on December 25, 2017.
  Fact 0.6: The 2018 NBA All-Star Game was played on February 18, 2018.
  Fact 0.7: The 2018 NBA All-Star Game was played at the Staples Center in Los Angeles, California.
  Fact 0.8: LeBron James of the Cleveland Cavaliers was named the All-Star Game Most Valuable Player.
  Fact 0.9: The 2017-18 NBA regular season ended on April 11, 2018.
  Fact 0.10: The 2017-18 NBA playoffs began on April 14, 2018.
Sentence 1: 2017 -- 18 NBA season League National Basketball Association Sport Basketball Duration October 17 , 2017 -- April 11 , 2018 April 14 -- May 28 , 2018 ( Playoffs ) May 31 -- June 17 , 2018 ( Finals ) Number of games 82 Number of teams 30 TV partner ( s ) ABC , TNT , ESPN , NBA TV Draft Top draft pick Markelle Fultz Picked by Philadelphia 76ers Regular season Top seed Houston Rockets Top scorer James Harden ( Houston ) Playoffs Finals NBA seasons ← 2016 -- 17 2018 -- 19 →
  Fact 1.0: The league of the 2017-18 NBA season is the National Basketball Association.
  Fact 1.1: The sport of the 2017-18 NBA season is basketball.
  Fact 1.2: The 2017-18 NBA regular season ran from October 17, 2017 to April 11, 2018.
  Fact 1.3: The 2017-18 NBA playoffs ran from April 14 to May 28, 2018.
  Fact 1.4: The 2017-18 NBA Finals ran from May 31 to June 17, 2018.
  Fact 1.5: The 2017-18 NBA season had 82 games.
  Fact 1.6: The 2017-18 NBA season had 30 teams.
  Fact 1.7: ABC, TNT, ESPN and NBA TV were the TV partners of the 2017-18 NBA season.
  Fact 1.8: Markelle Fultz was the top draft pick, picked by the Philadelphia 76ers.
  Fact 1.9: The Houston Rockets were the top seed in the 2017-18 NBA regular season.
  Fact 1.10: James Harden of the Houston Rockets was the top scorer of the 2017-18 NBA season.
  Fact 1.11: The 2016-17 season is the season before the 2017-18 NBA season.
  Fact 1.12: The 2018-19 season is the season after the 2017-18 NBA season.

Retrieved items:
<item 1>
The 2017–18 NBA regular season ran October 17, 2017–April 11, 2018; preseason included two Warriors–Timberwolves games in China. Milwaukee fired Jason Kidd in
January and appointed Joe Prunty interim coach. Toronto led the East with 59 wins. Leaders included James Harden in scoring and efficiency, Russell Westbrook in
assists and triple-doubles, and Andre Drummond in rebounds. LeBron James won four Eastern Player of the Month awards; Ben
</item 1>

<item 2>
The 2017–18 season was the NBA’s 72nd, featuring 30 teams playing 82 games from October 17, 2017, to April 11, 2018. Philadelphia selected Markelle Fultz first
in the draft. Houston earned the regular-season top seed, and James Harden led scoring. Playoffs ran April 14–May 28, followed by the Finals, May 31–June 17.
Games aired on ABC, TNT, ESPN, and
</item 2>

<item 3>
A partial NBA season index lists seasons from 1957–58 through 1994–95, grouped by decade from the 1960s to the 1990s. It also includes a heading for the 2017–18
NBA season.
</item 3>

<item 4>
2017–18 NBA season
The regular season began on October 17 , 2017 , earlier than previous seasons to reduce the number of `` back - to - back '' games teams are scheduled to play ,
with the 2017 Eastern Conference champion ( and Finals runner -- up ) Cleveland Cavaliers hosting a game against the Boston Celtics at Quicken Loans Arena in
Cleveland , Ohio Christmas games were played on December 25 , 2017 . The 2018 NBA All - Star Game was played on February 18 , 2018 , at the Staples Center in
Los Angeles , California . LeBron James of the Cleveland Cavaliers was named the All - Star Game Most Valuable Player . The regular season ended on April 11 ,
2018 and the playoffs began on April 14 , 2018 . Contents ( hide ) 1 Transactions 1.1 Retirement 1.2 In - season trades 1.3 Free agency 1.4 Coaching changes
1.4. 1 Off - season 1.4. 2 In - season 2 Preseason 2.1 International games 3 Regular season 3.1 By conference 3.1.
</item 4>

<item 5>
The NBA’s 72nd season ran October 17, 2017–April 11, 2018, with playoffs beginning April 14. Houston led the West (65–17), Toronto the East (59–23), and James
Harden topped scoring; LeBron James was All-Star MVP. Changes included Nike uniforms, corporate jersey patches, two-way contracts, revised timeout and
trade-deadline rules, and an earlier start to limit back-to-backs. Phoenix, Memphis, and Milwaukee
</item 5>

<item 6>
2017–18 NBA season
92 1992 -- 93 1993 -- 94 1994 -- 95 1995 -- 96 1996 -- 97 1997 -- 98 1998 -- 99 2000s 1999 -- 00 2000 -- 01 2001 -- 02 2002 -- 03 2003 -- 04 2004 -- 05 2005 --
06 2006 -- 07 2007 -- 08 2008 -- 09 2010s 2009 -- 10 2010 -- 11 2011 -- 12 2012 -- 13 2013 -- 14 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 2017 --
18 NBA season by team Draft Summer League All - Star Game Playoffs Finals Transactions G League Eastern Atlantic Boston Brooklyn New York Philadelphia Toronto
Central Chicago Cleveland Detroit Indiana Milwaukee Southeast Atlanta Charlotte Miami Orlando Washington Western Northwest Denver Minnesota Oklahoma City
Portland Utah Pacific Golden State L.A. Clippers L.A.
</item 6>

<item 7>
The 2017–18 NBA regular season began October 17, 2017, with Cleveland hosting Boston, and ended April 11, 2018; playoffs started April 14. The All-Star Game was
held February 18 in Los Angeles, with LeBron James named MVP. Free-agent signings began July 6. On February 8, Cleveland reshaped its roster, acquiring Jordan
Clarkson, Larry Nance Jr., Rodney Hood, and George Hill, while trading Isaiah Thomas, Channing
</item 7>

<item 8>
2017–18 NBA season
Africa 2017 Playoffs List Finals Champions History Predecessors BAA NBL ABA Merger Criticisms and controversies 2007 Tim Donaghy betting scandal Lockouts
Seasons Records regular season post-season All - Star Game Win - loss records Personalities Players Current rosters Foreign players Race and ethnicity First
overall draft picks Highest paid Retired numbers Banned or suspended NBPA Head coaches Current Player - coaches Champions Foreign coaches NBCA Owners Referees
Awards and honors Larry O'Brien Trophy NBA Awards NBA MVP Finals MVP All - Star Game MVP Hall of Fame Members NBA Silver Anniversary Team NBA 35th Anniversary
Team 50 Greatest Players Others Arenas Business Collective bargaining agreement Salary cap NBA Store Culture Cheerleading Mascots Dress code G League Midwest
Division Media TV NBA TV Rivalries Teams Defunct Expansion Relocated Timeline WNBA Basketball in the United States Category Portal 2017 -- 18 season Retrieved
from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
2017–18 NBA season
The regular season started eight days earlier than the previous season . This move by the NBA aims for fewer `` back - to - back '' games , even fewer than the
previous season which had already reached an all - time low to help address teams that are `` resting '' their superstars . The NBA decreased the maximum number
of timeouts in a regulation game from 18 to 14 . The three team timeouts during the last two minutes of the game were also decreased to two per team in the last
three minutes to improve the flow and pace of the game . The trade deadline was moved from four days after the NBA All - Star Game to ten days before it , so
teams can settle their rosters before the All - Star break . Free - throw shooters will be assessed a delay - of - game violation if they venture outside of the
three - point line in between attempts . The NBA also added two - way contracted players , signifying an increasing partnership between the NBA and the newly
branded NBA G League . Referees can now assess dangerous closeouts of defenders to jump shooters with little space to land for a technical or a flagrant foul .
</item 9>

<item 10>
2017–18 NBA season
2017 -- 18 NBA season - wikipedia 2017 -- 18 NBA season Jump to : navigation , search 2017 -- 18 NBA season League National Basketball Association Sport
Basketball Duration October 17 , 2017 -- April 11 , 2018 April 14 -- May 28 , 2018 ( Playoffs ) May 31 -- June 17 , 2018 ( Finals ) Number of games 82 Number of
teams 30 TV partner ( s ) ABC , TNT , ESPN , NBA TV Draft Top draft pick Markelle Fultz Picked by Philadelphia 76ers Regular season Top seed Houston Rockets Top
scorer James Harden ( Houston ) Playoffs Finals NBA seasons ← 2016 -- 17 2018 -- 19 → The 2017 -- 18 NBA season is the 72nd season of the National Basketball
Association ( NBA ) .
</item 10>
=== END ITEM id=8e802c1fc02664b8 ===

=== ITEM id=a19cbae1f7724681 ===
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
The Salton Sea is a shallow, saline, landlocked lake in California’s Imperial and Coachella valleys, accidentally created when the Colorado River breached
irrigation works in 1905. Its history includes earlier Lake Cahuilla cycles, military use, and resort communities. Reduced inflows from drought and water
transfers have raised salinity and worsened algal blooms, fish kills, exposed-bed dust, and wildlife habitat loss, despite its importance to hundreds of bird
species and the Pacific Flyway. Research also
</item 1>

<item 2>
The Salton Sea’s salinity is about 56 g/L—higher than the Pacific Ocean’s 35 g/L but within the Great Salt Lake’s 50–270 g/L range. Salinity has been rising
roughly 3% annually, while about 4 million short tons of salt enter the valley each year. The excerpt lists sections on the sea’s history, ecology, fish and
bird populations, increasing salinity, and remediation proposals.
</item 2>

<item 3>
Salton Sea
Salton Sea - wikipedia Salton Sea Jump to : navigation , search For other uses , see Salton Sea ( disambiguation ) . Salton Sea Location Colorado Desert
Imperial and Riverside Counties , California , U.S. Coordinates 33 ° 18 ′ N 115 ° 48 ′ W ﻿ / ﻿ 33.3 ° N 115.8 ° W ﻿ / 33.3 ; - 115.8 Coordinates : 33 ° 18 ′ N
115 ° 48 ′ W ﻿ / ﻿ 33.3 ° N 115.8 ° W ﻿ / 33.3 ; - 115.8 Type Endorheic rift lake Primary inflows Alamo River New River Whitewater River Primary outflows None
Catchment area 8,360 square miles ( 21,700 km ) Basin countries United States , Mexico Surface area 343 sq mi ( 889 km ) Max .
</item 3>

<item 4>
The Salton Sea is a shallow, saline, endorheic rift lake on the San Andreas Fault in California’s Imperial and Coachella valleys. In the Colorado Desert, it
lies 236 feet below sea level (January 2018), reaches 43 feet deep, and holds about 6 million acre-feet of water. It is fed by the New, Whitewater, and Alamo
rivers, agricultural runoff, drainage systems, and creeks. Nearby settlements include Bombay Beach, Desert Beach
</item 4>

<item 5>
Salton Sea
The Salton Sea is also a major resting stop on the Pacific Flyway . On 18 November 2006 , a Ross 's gull , a high Arctic bird , was sighted and photographed
there . Increasing salinity ( edit ) Dead fish on the western shore of Salton City The lack of an outflow means the Salton Sea does not have a natural
stabilization system ; it is very dynamic . Fluctuations in the water level caused by variations in agricultural runoff , the ancient salt deposits in the lake
bed , and the relatively high salinity of the inflow feeding the sea are all causing ever - increasing salinity . The body was initially a freshwater lake , but
by the 1960s its rising salinity had begun to jeopardize some of the species in it . With a salinity now exceeding 5.0 % w / v ( saltier than seawater ) , most
species of fish can no longer survive there . A freshwater fish notable for its ability to withstand the rising salinity of the Salton Sea , the desert pupfish
, can survive salinities ranging from 0.0 % to 7.0 % .
</item 5>

<item 6>
Salton Sea
Ecology ( edit ) See also : Sonny Bono Salton Sea National Wildlife Refuge § Wildlife Fish population ( edit ) Due to the high salinity , very few fish species
can tolerate living in the Salton Sea . Introduced tilapia are the main fish that can tolerate the high salinity levels and pollution . Other freshwater fish
species live in the rivers and canals that feed the Salton Sea , including threadfin shad , carp , red shiner , channel catfish , white catfish , largemouth
bass , mosquitofish , sailfin molly , and the vulnerable desert pupfish . Avian population ( edit ) The Salton Sea has been termed a `` crown jewel of avian
biodiversity '' by Dr. Milt Friend of the Salton Sea Science Office . Over 400 species have been documented at the Salton Sea . The most diverse and probably
the most significant populations of bird life in the continental United States are hosted , rivaled only by Big Bend National Park in Texas . It supports 30 %
of the remaining population of the American white pelican . The Salton Sea is also a major resting stop on the Pacific Flyway .
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
Colorado River flooding submerged Salton, railroad facilities, and Torres-Martinez lands, forming the undrained Salton Sea. A Navy test base operated there from
1942 to 1978. Mid-century resorts later declined as agricultural runoff increased salinity and pollution. Eutrophication, fish kills, exposed lakebed, dust, and
disrupted bird migration worsened with reduced inflow, especially after water transfers. Restoration proposals since 1955 include impoundments, wetlands, and
importing
</item 8>

<item 9>
Salton Sea
Surface water temperature ( Sandy Beach ) Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec 57 ° F 14 ° C 57 ° F 14 ° C 64 ° F 18 ° C 72 ° F 22 ° C 77 ° F 25 ° C
82 ° F 28 ° C 90 ° F 32 ° C 91 ° F 33 ° C 86 ° F 30 ° C 81 ° F 27 ° C 72 ° F 22 ° C 61 ° F 17 ° C In popular culture ( edit ) An abandoned boat stuck in the
ground , close to the west coast marina of the Salton Sea Films The documentary Plagues & Pleasures on the Salton Sea ( which premiered at the 2004 Slamdance
Film Festival and , with the addition of John Waters as narrator , premiered at the Provincetown International Film Festival in Massachusetts and opened in
select theaters on February 24 , 2006 ) , covers the first 100 years of the Salton Sea , along with the environmental issues and offbeat residents of the region
.
</item 9>

<item 10>
The Salton Sea’s lack of an outlet and changing agricultural inflows have driven salinity above 5%, while pollution and algal blooms have caused mass fish
die-offs. Introduced tilapia and salt-tolerant desert pupfish are among the few fish able to survive. Despite its ecological stress, the sea supports more than
400 bird species, hosts about 30% of the remaining American white pelican population, and is a major Pacific Flyway stop. Fish carcasses and the pervasive,
</item 10>
=== END ITEM id=a19cbae1f7724681 ===

=== ITEM id=434a0fa55584e394 ===
Gold sentences and facts:
Sentence 0: The American entry into World War I came in April 1917 , after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war . Apart from an Anglophile element urging early support for the British , American public opinion reflected that of the president : the sentiment for neutrality was particularly strong among Irish Americans , German Americans and Scandinavian Americans , as well as among church leaders and among women in general . On the other hand , even before World War I had broken out , American opinion had been more negative toward Germany than towards any other country in Europe . Over time , especially after reports of atrocities in Belgium in 1914 and following the sinking of the passenger liner RMS Lusitania in 1915 , the American people increasingly came to see Germany as the aggressor in Europe .
  Fact 0.0: The American entry into World War I came in April 1917.
  Fact 0.1: The American entry into World War I came after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war.
  Fact 0.2: Apart from an Anglophile element urging early support for the British, American public opinion reflected that of the president.
  Fact 0.3: The sentiment for neutrality was particularly strong among Irish Americans.
  Fact 0.4: The sentiment for neutrality was particularly strong among German Americans.
  Fact 0.5: The sentiment for neutrality was particularly strong among Scandinavian Americans.
  Fact 0.6: The sentiment for neutrality was particularly strong among church leaders.
  Fact 0.7: The sentiment for neutrality was particularly strong among women in general.
  Fact 0.8: Before World War I had broken out, American opinion had been more negative toward Germany than toward any other country in Europe.
  Fact 0.9: Over time, the American people increasingly came to see Germany as the aggressor in Europe.
  Fact 0.10: The shift toward seeing Germany as the aggressor was especially driven by reports of atrocities in Belgium in 1914.
  Fact 0.11: The shift toward seeing Germany as the aggressor was especially driven by the sinking of the passenger liner RMS Lusitania in 1915.

Retrieved items:
<item 1>
American entry into World War I
However , they overestimated how many ships they could sink and thus the extent Britain would be weakened . Finally , they did not foresee that convoys could
and would be used to defeat their efforts . They believed that the United States was so weak militarily that it could not be a factor on the Western Front for
more than a year . The civilian government in Berlin objected , but the Kaiser sided with his military . Business considerations ( edit ) The beginning of war
in Europe coincided with the end of the Recession of 1913 -- 1914 in America . Exports to belligerent nations rose rapidly over the first four years of the War
from $824.8 million in 1913 to $2.25 billion in 1917 . Loans from American financial institutions to the Allied nations in Europe also increased dramatically
over the same period . Economic activity towards the end of this period boomed as government resources aided the production of the private sector . Between 1914
and 1917 , industrial production increased 32 % and GNP increased by almost 20 % . The improvements to industrial production in the United States outlasted the
war .
</item 1>

<item 2>
American entry into World War I
The solution was a third route , a `` peace without victory '' , according to Wilson . On April 6 , 1917 , Congress declared war . In the Senate , the
resolution passed 82 to 6 , with Senators Harry Lane , William J. Stone , James Vardaman , Asle Gronna , Robert M. La Follette , Sr. , and George W. Norris
voting against it . In the House , the declaration passed 373 to 50 , with Claude Kitchin , a senior Democrat , notably opposing it . Another opponent was
Jeannette Rankin , who alone voted against entry into both World War I and World War II . Nearly all of the opposition came from the West and the Midwest .
Austria - Hungary ( edit ) The United States Senate , in a 74 to 0 vote , declared war on Austria - Hungary on December 7 , 1917 , citing Austria - Hungary 's
severing of diplomatic relations with the United States , its use of unrestricted submarine warfare and its alliance with Germany . The declaration passed in
the United States House of Representatives by a vote of 365 to 1 .
</item 2>

<item 3>
American entry into World War I
As U.S. president , it was Wilson who made the key policy decisions over foreign affairs : while the country was at peace , the domestic economy ran on a
laissez - faire basis , with American banks making huge loans to Britain and France -- funds that were in large part used to buy munitions , raw materials , and
food from across the Atlantic . Until 1917 , Wilson made minimal preparations for a land war and kept the United States Army on a small peacetime footing ,
despite increasing demands for enhanced preparedness . He did however expand the United States Navy . In 1917 , with Russia experiencing political upheaval
following widespread disillusionment there over the war , and with Britain and France low on credit , Germany appeared to have the upper hand in Europe , while
Germany 's ally , the Ottoman Empire , clung stubbornly to its possessions in the Middle East . In the same year , Germany decided to resume unrestricted
submarine warfare against any vessel approaching British waters ; this attempt to starve Britain into surrender was balanced against the knowledge that it would
almost certainly bring the United States into the war .
</item 3>

<item 4>
American entry into World War I
Sinking of American merchant ships ( edit ) In early 1917 Berlin forced the issue . Its declared decision on 31 January 1917 to target neutral shipping in a
designated war - zone became the immediate cause of the entry of the United States into the war . Five American merchant ships went down in March . Outraged
public opinion now overwhelmingly supported Wilson when he asked Congress for a declaration of war on April 2 , 1917 . Public opinion , moralism , and national
interest ( edit ) Historians such as Ernest R. May have approached the process of American entry into the war as a study in how public opinion changed radically
in three years ' time . In 1914 most Americans called for neutrality , seeing the war a dreadful mistake and were determined to stay out . By 1917 the same
public felt just as strongly that going to war was both necessary and wise . Military leaders had little to say during this debate , and military considerations
were seldom raised . The decisive questions dealt with morality and visions of the future . The prevailing attitude was that America possessed a superior moral
position as the only great nation devoted to the principles of freedom and democracy .
</item 4>

<item 5>
American entry into World War I
International Encyclopedia of the First World War . NY Times main headline , April 2 , 1917 , President Calls for War Declaration , Stronger Navy , New Army of
500,000 Men , Full Cooperation With Germany 's Foes President Wilson 's War Address Map of Europe at the time of the US declaration of war on Germany at
omniatlas.com World War I : Declarations of War from Around the Globe - How America Entered the Great War Today in History : U.S.
</item 5>

<item 6>
American entry into World War I
252 -- 282 . ^ Jump up to : H.J. Res. 169 : Declaration of War with Austria - Hungary , WWI , United States Senate ^ Jump up to : Jennifer K. Elsea ; Matthew C.
Weed ( April 18 , 2014 ) . `` Declarations of War and Authorizations for the Use of Military Force : Historical Background and Legal Implications '' ( PDF ) .
p. 9 . Archived from the original ( PDF ) on August 10 , 2006 . Retrieved July 21 , 2014 . Jump up ^ Eric W. Osborne , Britain 's economic blockade of Germany ,
1914 -- 1919 ( 2004 ) Jump up ^ Ernest May , The World War and American Isolation , 1914 -- 1917 ( 1959 ) p. 115 quote from Dec 1914 . Jump up ^ C.R.M.F.
Cruttwell , A history of the Great War , 1914 -- 1918 ( 2007 ) p. 191 Jump up ^ Edward House , The intimate papers of Colonel House : Vol 2 ( 1928 ) p.
</item 6>

<item 7>
American entry into World War I
Germany 's U-boat warfare was paramount . Had it not been for Berlin 's announcement of January 31 , the president probably would not have issued a call to arms
. Jump up ^ Justus D. Doenecke , Nothing Less Than War : A New History of America 's Entry into World War I ( 2011 ) ch 10 ^ Jump up to : May , Ernest R. ( 1966
) . The World War and American isolation , 1914 -- 1917 . Jump up ^ Knock , Thomas J. ( 1995 ) . To End All Wars : Woodrow Wilson and the Quest for a New World
Order . Jump up ^ Syrett , Harold C. ( 1945 ) . `` The Business Press and American Neutrality , 1914 -- 1917 '' . Mississippi Valley Historical Review . 32 ( 2
) : 215 -- 230 . doi : 10.2307 / 1898209 . JSTOR 1898209 .
</item 7>

<item 8>
American entry into World War I
However , after the US did join the war in April , 1917 a schism developed between the anti-war Party majority and a pro-war faction of Socialist writers ,
journalists and intellectuals led by John Spargo , William English Walling and E. Haldeman - Julius . This group founded the rival Social Democratic League of
America to promote the war effort among their fellow Socialists . Next were the more moderate Liberal - Internationalists . This bipartisan group reluctantly
supported a declaration of war against Germany with the postwar goal of establishing collective international security institutions designed to peacefully
resolve future conflicts between nations and to promote liberal democratic values more broadly . This groups 's views were advocated by interest groups such as
the League to Enforce Peace . Adherents included US President Woodrow Wilson , his influential advisor Edward M. House , former President William Howard Taft ,
famed inventor Alexander Graham Bell , Wall Street financier Bernard Baruch and Harvard University President Abbott Lawrence Lowell . Finally , there were the
so - called Atlanticists . Ardently pro-Entente , they had strongly championed American intervention in the war since 1915 .
</item 8>

<item 9>
American entry into World War I
See also ( edit ) World War I portal Causes of World War I Diplomatic history of World War I United States in World War I United States home front during World
War I Footnotes ( edit ) Jump up ^ Jeanette Keith ( 2004 ) . Rich Man 's War , Poor Man 's Fight : Race , Class , and Power in the Rural South during the First
World War . U. of North Carolina Press . pp. 1 -- 5 . ISBN 978 - 0 - 8078 - 7589 - 6 . Jump up ^ Barnes , Harry Elmer . The Genesis of the World War ( 1925 )
pp. 590 -- 591 Jump up ^ `` World War One '' . BBC History . Jump up ^ Link , Arthur S. ( 1972 ) . Woodrow Wilson and the Progressive Era , 1910 -- 1917 . New
York : Harper & Row . pp. 252 -- 282 . ^ Jump up to : H.J. Res.
</item 9>

<item 10>
American entry into World War I
American entry into World war I - wikipedia American entry into World war I Jump to : navigation , search The American entry into World War I came in April 1917
, after more than two and a half years of efforts by President Woodrow Wilson to keep the United States out of the war . Apart from an Anglophile element urging
early support for the British , American public opinion reflected that of the president : the sentiment for neutrality was particularly strong among Irish
Americans , German Americans and Scandinavian Americans , as well as among church leaders and among women in general . On the other hand , even before World War
I had broken out , American opinion had been more negative toward Germany than towards any other country in Europe . Over time , especially after reports of
atrocities in Belgium in 1914 and following the sinking of the passenger liner RMS Lusitania in 1915 , the American people increasingly came to see Germany as
the aggressor in Europe . As U.S.
</item 10>
=== END ITEM id=434a0fa55584e394 ===

=== ITEM id=0543f941953938b7 ===
Gold sentences and facts:
Sentence 0: Principal photography started April 26 , 1965 , at the 20th Century - Fox Studios and 20th Century - Fox Ranch , California . Other filming locations , simulating the desert , were Buttercup Valley , Arizona and Pilot Knob Mesa , California . The flying sequences were all filmed at Pilot Knob Mesa near Winterhaven , located in California 's Imperial Valley , on the western fringes of Yuma , Arizona .
  Fact 0.0: Principal photography started on April 26, 1965.
  Fact 0.1: Principal photography started at the 20th Century-Fox Studios in California.
  Fact 0.2: Principal photography started at the 20th Century-Fox Ranch in California.
  Fact 0.3: Other filming locations, which simulated the desert, were Buttercup Valley in Arizona.
  Fact 0.4: Other filming locations, which simulated the desert, were Pilot Knob Mesa in California.
  Fact 0.5: The flying sequences were all filmed at Pilot Knob Mesa.
  Fact 0.6: Pilot Knob Mesa is near Winterhaven.
  Fact 0.7: Pilot Knob Mesa is located in California's Imperial Valley.
  Fact 0.8: Pilot Knob Mesa is on the western fringes of Yuma, Arizona.

Retrieved items:
<item 1>
The Flight of the Phoenix (1965 film)
Dorfmann claims that the principles are exactly the same , and that in many aspects models require much more exacting designs and can be less forgiving than
full - size aircraft , but Towns and Moran are horrified at the idea of flying a plane made by a man who works with `` toys '' . Without any other choice ,
however , they decide to forge ahead with the plan . Just as the water supplies are exhausted , the Phoenix is completed . Only seven starter cartridges are
available for the engine , and the first four startup attempts are unsuccessful . Towns decides to fire the fifth cartridge with the ignition off , to clear the
engine 's cylinders , which he does over Dorfmann 's strenuous objection . The next startup is successful . The men pull the Phoenix to a nearby hilltop and
climb onto the wings . When Towns guns the engine , the Phoenix slides down the hill and along a lake bed before taking off . After landing at an oasis with a
manned oil rig , the men celebrate and Towns and Dorfmann are reconciled . Cast ( edit ) James Stewart as Capt .
</item 1>

<item 2>
The Flight of the Phoenix (1965 film)
Cast ( edit ) James Stewart as Capt . Frank Towns Richard Attenborough as Lew Moran Hardy Krüger as Heinrich Dorfmann Peter Finch as Capt . Harris Ernest
Borgnine as Trucker Cobb Ian Bannen as `` Ratbags '' Crow Ronald Fraser as Sgt . Watson Christian Marquand as Dr. Renaud Dan Duryea as Standish George Kennedy
as Mike Bellamy Gabriele Tinti as Gabriel Alex Montoya as Carlos Peter Bravos as Tasso William Aldrich as Bill Barrie Chase as Farida Stanley Ralph Ross as Arab
singer ( voice ; uncredited ) Production ( edit ) This section needs additional citations for verification . Please help improve this article by adding
citations to reliable sources . Unsourced material may be challenged and removed . ( January 2017 ) ( Learn how and when to remove this template message )
Locations ( edit ) Principal photography started April 26 , 1965 , at the 20th Century - Fox Studios and 20th Century - Fox Ranch , California . Other filming
locations , simulating the desert , were Buttercup Valley , Arizona and Pilot Knob Mesa , California .
</item 2>

<item 3>
The Flight of the Phoenix (1965 film)
Although principal photography was completed on August 13 , 1965 , in order to complete filming , a North American O - 47 A ( N4725V ) from the Planes of Fame
Air Museum in Chino , California was modified and used as a flying Phoenix stand - in . With the canopy removed , a set of skids attached to the main landing
gear as well as ventral fin added to the tail , it sufficed as more - or-less a visual lookalike . Filming using the O - 47A was completed in October / November
1965 . It appears in the last flying scenes , painted to look like the earlier Phoenix P - 1 . The final production utilized a mix of footage that included the
O - 47A , the `` cobbled - together '' Phoenix and Phoenix P - 1 . The final credit on the screen was , `` It should be remembered ... that Paul Mantz , a fine
man and a brilliant flyer gave his life in the making of this film ... '' Reception ( edit ) The film opened in selected theaters on December 15 , 1965 , with a
full release in 1966 .
</item 3>

<item 4>
The Flight of the Phoenix (1965 film)
The story describes a small group of men struggling to survive their aircraft 's emergency landing in the Sahara Desert , and stars Richard Attenborough , Peter
Finch , Hardy Krüger and Ernest Borgnine . The ensemble cast includes Ian Bannen , Ronald Fraser , Christian Marquand , Dan Duryea and George Kennedy as other
passengers on the aircraft . Though the film was a failure at the box office , it has since gained a large following . Contents ( hide ) 1 Plot 2 Cast 3
Production 3.1 Locations 3.2 Aircraft used 4 Reception 5 Awards 6 See also 7 References 8 Further reading 9 External links Plot ( edit ) Frank Towns ( James
Stewart ) is the pilot of a twin - engine Fairchild C - 82 Packet cargo plane flying from Jaghbub to Benghazi in Libya ; Lew Moran ( Richard Attenborough ) is
the navigator . The passengers include Capt . Harris ( Peter Finch ) and Sgt . Watson ( Ronald Fraser ) of the British Army ; Dr.
</item 4>

<item 5>
The Flight of the Phoenix (1965 film)
The R4Q - 1 was purchased from Allied Aircraft of Phoenix , Arizona . The aerial camera platform was a B - 25J Mitchell , N1042B , which was also used in the
1970 film Catch - 22 . The flying sequences were flown by Paul Mantz , co-owner of the Tallmantz Aviation , filling in for his partner Frank Tallman , who had
injured his leg . A famous racing / stunt / movie pilot and collector of warplanes , Paul Mantz was flying the Tallmantz Phoenix P - 1 , the machine that was ``
made of the wreckage '' , in front of the cameras on the morning of July 8 , 1965 . He was performing touch - and - go landings , and on one touchdown the
fuselage buckled . The movie model broke apart and cartwheeled , killing Mantz and seriously injuring stuntman Bobby Rose on board .
</item 5>

<item 6>
The Flight of the Phoenix (1965 film)
Harris and Carlos attempt to walk to an oasis . Carlos leaves his monkey behind with the men . Harris and Towns refuse to let Cobb go along , but Cobb defiantly
follows and dies . Days later , Harris returns to the crash site alone and barely alive . Meanwhile , Dorfmann has been working on a radical idea : He believes
they can build a new aircraft from the wreckage . The C - 82 has twin booms extending rearwards from each engine and connected by the horizontal stabilizer .
Dorfmann 's plan is to attach the outer sections of the both wings to the left engine and left boom , discarding the center fuselage and both inner wing
sections of the aircraft . The passengers will ride on top of the wings . Harris and Moran believe he is either joking or delusional . The argument is
complicated by a personality clash between Towns , a proud old traditionalist aviator , and Dorfmann , an equally proud young technician . Moran struggles to
keep the peace .
</item 6>

<item 7>
The Flight of the Phoenix (1965 film)
( January 2017 ) ( Learn how and when to remove this template message ) The Flight of the Phoenix 1965 theatrical release poster Directed by Robert Aldrich
Produced by Robert Aldrich Written by Lukas Heller Based on The Flight of the Phoenix by Elleston Trevor Starring James Stewart Richard Attenborough Hardy
Krüger Peter Finch Ernest Borgnine Ian Bannen Music by Frank De Vol Cinematography Joseph Biroc Edited by Michael Luciano Production company The Associates and
Aldrich Distributed by 20th Century Fox Release date December 15 , 1965 ( 1965 - 12 - 15 ) Running time 142 minutes Country United States Language English
Budget $5,355,000 or $3.8 million Box office $3 million ( US / Canada rentals ) 311,136 admissions ( France ) The Flight of the Phoenix is a 1965 American drama
film starring James Stewart , produced and directed by Robert Aldrich , and based on the 1964 novel The Flight of the Phoenix by Elleston Trevor .
</item 7>

<item 8>
The Flight of the Phoenix (1965 film)
The flying sequences were all filmed at Pilot Knob Mesa near Winterhaven , located in California 's Imperial Valley , on the western fringes of Yuma , Arizona .
Aircraft used ( edit ) The Phoenix `` static '' model as seen in the film In 2005 , Hollywood aviation historian Simon Beck identified the aircraft used in the
film : Fairchild C - 82A Packet , N6887C -- flying shots . Fairchild C - 82A Packet , N4833V -- outdoor location wreck . Fairchild C - 82A Packet , N53228 --
indoor studio wreck . Fairchild R4Q - 1 Flying Boxcar ( the USMC C - 119C variant ) , BuNo. 126580 -- non-flying Phoenix prop . Tallmantz Phoenix P - 1 , N93082
-- flying Phoenix aircraft . North American O - 47 A , N4725V -- second flying Phoenix . The C - 82As were from Steward - Davies Inc. at Long Beach , California
, while the O - 47A came from the Planes of Fame air museum in California .
</item 8>

<item 9>
The Flight of the Phoenix (1965 film)
Bosley Crowther of the New York Times dismissed it as `` grim and implausible '' , while Variety praised the film as an `` often - fascinating and superlative
piece of filmmaking highlighted by standout performances and touches that show producer - director at his best . '' Awards ( edit ) The Flight of the Phoenix
was nominated for two Academy Awards : Ian Bannen for Supporting Actor and Michael Luciano for Film Editing . See also ( edit ) Flight of the Phoenix ( 2004
film ) , a re-make of this film Lady Be Good ( aircraft ) List of American films of 1965 List of film accidents Survival film , about the film genre , with a
list of related films Tragedy at Kufra References ( edit ) ^ Jump up to : Solomon , Aubrey ( 2002 ) . Twentieth Century Fox : A Corporate and Financial History
( 1st ed . ) . Lanham , Maryland : Scarecrow . ISBN 9780810842441 . Jump up ^ Silver , Alain ; Ursini , James ( 1995 ) . Whatever Happened to Robert Aldrich ?
</item 9>

<item 10>
The Flight of the Phoenix (1965 film)
Watson ( Ronald Fraser ) of the British Army ; Dr. Renaud ( Christian Marquand ) , a physician ; Heinrich Dorfmann ( Hardy Krüger ) , a German aeronautical
engineer ; and an oil company accountant named Standish ( Dan Duryea ) . There are also several oil workers , including Trucker Cobb ( Ernest Borgnine ) , a
mentally - disturbed foreman ; Ratbags Crow ( Ian Bannen ) , a mean - spirited , sardonic Scot ; Carlos ( Alex Montoya ) and his pet monkey ; and Gabriel (
Gabriele Tinti ) . A sudden sandstorm disables the engines , forcing Towns to crash - land in the desert . As the aircraft careens to a stop , several oil drums
and oil drilling tools break loose and severely injure Gabriel 's leg . Two other workers are killed . The radio is unusable , but the survivors are too far off
- course to be found and rescued . They have a large quantity of dates for food , but their water will last for only ten to fifteen days even if they avoid
physical exertion .
</item 10>
=== END ITEM id=0543f941953938b7 ===

=== ITEM id=19c3f0379b94e7e9 ===
Gold sentences and facts:
Sentence 0: Ballets by Pyotr Ilyich Tchaikovsky Swan Lake ( 1876 ) Sleeping Beauty ( 1889 ) The Nutcracker ( 1892 ) List of all compositions
  Fact 0.0: The list is titled Ballets by Pyotr Ilyich Tchaikovsky.
  Fact 0.1: Swan Lake, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1876.
  Fact 0.2: Sleeping Beauty, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1889.
  Fact 0.3: The Nutcracker, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1892.
  Fact 0.4: The text contains the label "List of all compositions".

Retrieved items:
<item 1>
The Sleeping Beauty (ballet)
* 1895 ) The Sleeping Beauty ( 1890 ) Nénuphar ( 1890 ) Kalkabrino ( 1891 ) A Fairy Tale ( 1891 ) La Sylphide ( * 1892 ) The Nutcracker ( 1892 ) Cinderella (
1893 ) The Awakening of Flora ( 1894 ) Swan Lake ( * 1895 ) The Little Humpbacked Horse ( as `` The Tsar Maiden '' ) ( * 1895 ) The Calvary Halt ( 1896 ) The
Pearl ( 1896 ) Bluebeard ( 1896 ) Raymonda ( 1898 ) 1900 -- 03 Les Ruses d'Amour or The Trial of Damis ( 1900 ) The Seasons ( 1900 ) Harlequinade ( 1900 ) The
Heart of the Marquis ( 1902 ) The Magic Mirror ( 1903 ) The Romance of the Rosebud and the Butterfly ( never presented ) Pyotr Ilyich Tchaikovsky Operas The
Voyevoda Undina The Oprichnik Vakula the Smith Eugene Onegin The Maid of Orleans Mazepa
</item 1>

<item 2>
The Sleeping Beauty (ballet)
This version has been performed by a number of central European ballet companies over the past two decades . Roles and original cast ( edit ) Role Mariinsky
1890 Bolshoi 1899 King Florestan Feliks Krzesiński ( Mathilde Kschessinska 's father ) Queen Giuseppina Cecchetti Princess Aurora , the Sleeping Beauty Carlotta
Brianza Lyubov Roslavleva Lilac Fairy Marie Petipa M. Grachevskaya Carabosse Enrico Cecchetti Vasiliy Geltser Prince Désiré Pavel Gerdt Ivan Khlyustin Bluebird
Enrico Cecchetti Princess Florine Varvara Nikitina The ballet 's premiere received more favorable accolades than Swan Lake from the press but Tchaikovsky never
had the luxury of being able to witness his work become an instant success in theatres outside of Russia . He died in 1893 .
</item 2>

<item 3>
The Sleeping Beauty (ballet)
* 1896 ) 1880 -- 99 La Fille du Danube ( * 1880 ) Zoraiya ( 1881 ) La Vivandière ( as `` Markitenka '' ) ( * 1881 ) Pâquerette ( * 1882 ) The Night and the Day
( 1883 ) Pygmalion ( 1883 ) Coppélia ( * 1884 ) Giselle ( * 1884 , 1899 , 1903 ) Le Diable à Quatre ( as `` The Willful Wife '' ) ( * 1885 ) La Fille Mal Gardée
( * 1885 ) The Magic Pills ( 1886 ) The King 's Command ( 1886 , * 1887 , * 1900 ) La Esmeralda ( * 1886 , * 1899 ) The Sacrifices to Cupid ( 1886 ) * Fiametta
( 1887 ) The Vestal ( 1888 ) The Talisman ( 1889 , * 1895 ) The Enchanted Forest ( * 1889 ) The Caprices of a Butterfly ( 1889 , *
</item 3>

<item 4>
The Sleeping Beauty (ballet)
* 1891 , * 1903 ) Don Quixote ( 1869 , * 1871 ) Trilby ( 1870 ) Catarina ( * 1870 ) The Two Stars ( 1871 ) La Péri ( * 1872 ) Camargo ( 1872 ) Le Papillon ( *
1874 ) Ondine ( as `` The Naiad and the Fisherman '' ) ( * 1874 , * 1892 ) The Bandits ( 1875 ) The Adventures of Peleus ( 1876 , * 1897 ) A Midsummer Night 's
Dream ( 1876 ) La Bayadère ( 1877 , * 1900 ) Roxana ( 1878 ) Ariadne ( 1878 ) The Daughter of the Snows ( 1879 ) Frizak the Barber ( 1879 ) Mlada ( 1879 ,
</item 4>

<item 5>
The Sleeping Beauty (ballet)
The work has become one of the classical repertoire 's most famous ballets . Contents ( hide ) 1 History 1.1 Composition history 1.2 Performance history 2 Roles
and original cast 3 Instrumentation 4 Roles 5 Synopsis 6 Structure 7 Versions by other hands 7.1 Piano arrangements 7.2 Aurora 's Wedding by Sergei Diaghilev 8
Trademark controversy 9 References 10 External links 10.1 Video samples 10.2 Scores 10.3 History History ( edit ) Composition history ( edit ) Tchaikovsky was
approached by the Director of the Imperial Theatres in St. Petersburg , Ivan Vsevolozhsky on 25 May 1888 about a possible ballet adaptation on the subject of
the story of Undine . It was later decided that Charles Perrault 's La Belle au bois dormant would be the story for which Tchaikovsky would compose the music
for the ballet . Tchaikovsky did not hesitate to accept the commission , although he was aware that his only previous ballet , Swan Lake , met with little
enthusiasm at that stage of his career .
</item 5>

<item 6>
The Sleeping Beauty (ballet)
The ballet scenario that Tchaikovsky worked on was based on the Brothers Grimm 's version of Perrault 's work entitled ' Dornröschen ' . In that version , the
Princess 's parents ( the King and the Queen ) survived the 100 - year sleep to celebrate the Princess 's wedding to the Prince . However , Vsevolozhsky
incorporated Perrault 's other characters from his stories into the ballet , such as Puss in Boots , Little Red Riding Hood , Cinderella , Bluebird , Bluebeard
, Ricky of the Tuft and Tom Thumb . Other French fairy tale characters to be featured are Beauty and the Beast , Pretty Goldilocks and The White Cat .
Regardless , Tchaikovsky was happy to inform the Director of the Imperial Theatre that he had great pleasure studying the work and had come away with adequate
inspiration to do it justice . The choreographer was Marius Petipa , ballet master of the Imperial Ballet , who wrote a very detailed list of instructions as to
the musical requirements .
</item 6>

<item 7>
The Sleeping Beauty (ballet)
The Sleeping Beauty ( ballet ) - wikipedia The Sleeping Beauty ( ballet ) Jump to : navigation , search Ballets by Pyotr Ilyich Tchaikovsky Swan Lake ( 1876 )
Sleeping Beauty ( 1889 ) The Nutcracker ( 1892 ) List of all compositions The Sleeping Beauty ( Russian : Спящая красавица / Spyashchaya krasavitsa ) is a
ballet in a prologue and three acts , first performed in 1890 . The music was composed by Pyotr Ilyich Tchaikovsky ( his opus 66 ) . The score was completed in
1889 , and is the second of his three ballets . The original scenario was conceived by Ivan Vsevolozhsky , and is based on Charles Perrault 's La Belle au bois
dormant . The choreographer of the original production was Marius Petipa . The premiere performance took place at the Mariinsky Theatre in St. Petersburg on
January 15 , 1890 . The work has become one of the classical repertoire 's most famous ballets .
</item 7>

<item 8>
The Sleeping Beauty (ballet)
The Oprichnik Vakula the Smith Eugene Onegin The Maid of Orleans Mazepa ( or Mazeppa ) Cherevichki The Enchantress The Queen of Spades Iolanta Ballets Swan Lake
The Sleeping Beauty The Nutcracker Symphonies No. 1 in G minor ( `` Winter Daydreams '' ) No. 2 in C minor ( `` Little Russian '' ) No. 3 in D major ( `` Polish
'' ) No. 4 in F minor Symphony in B minor ( `` Manfred '' ) No. 5 in E minor No. 6 in B minor ( `` Pathétique '' ) No. 7 in E ♭ major ( unfinished ) Concertante
Piano Concerto No. 1 in B ♭ minor Piano Concerto No. 2 in G major Piano Concerto No.
</item 8>

<item 9>
The Sleeping Beauty (ballet)
1 in D major String Quartet No. 2 in F major String Quartet No. 3 in E ♭ minor Souvenir d'un lieu cher Piano Trio in A minor String Sextet in D minor ( Souvenir
de Florence ) Piano music Scherzo à la russe Souvenir de Hapsal The Seasons ( Les saisons ) Piano Sonata No.
</item 9>

<item 10>
The Sleeping Beauty (ballet)
With subtitles in Esperanto . Solo of Prince Désiré from Pas de deux ( Act 3 ) danced by Rudolf Nureyev ( from YouTube ) Scores ( edit ) Sleeping Beauty :
Scores at the International Music Score Library Project ( IMSLP ) History ( edit ) Tchaikovsky Research Ballets by Pyotr Ilyich Tchaikovsky Swan Lake , Op. 20
Sleeping Beauty , Op. 66 The Nutcracker , Op. 71 List of compositions by Pyotr Ilyich Tchaikovsky The ballets and * revivals of Marius Petipa in Russia 1847 --
59 Paquita ( * 1847 , * 1881 ) Le Diable amoureux ( as `` Satanella '' ) ( * 1848 ) Leda , the Swiss Milkmaid ( 1849 ) Giselle ( * 1850 , * 1884 , * 1899 , *
1903 ) The Star of Granada ( 1855 ) The Rose , the Violet , and the Butterfly ( 1857 ) Le Corsaire ( * 1858 , * 1863 , * 1868 ,
</item 10>
=== END ITEM id=19c3f0379b94e7e9 ===

=== ITEM id=8a84eac959c868f8 ===
Gold sentences and facts:
Sentence 0: The Homestead Acts were several United States federal laws that gave an applicant ownership of land , typically called a `` homestead , '' at no cost . In all , more than 270 million acres of public land , or nearly 10 % of the total area of the U.S. , was given away free to 1.6 million homesteaders ; most of the homesteads were west of the Mississippi River .
  Fact 0.0: The Homestead Acts were several United States federal laws.
  Fact 0.1: The Homestead Acts gave an applicant ownership of land.
  Fact 0.2: The land given under the Homestead Acts is typically called a homestead.
  Fact 0.3: The land was given at no cost.
  Fact 0.4: More than 270 million acres of public land were given away free under the Homestead Acts.
  Fact 0.5: The 270 million acres of public land were nearly 10% of the total area of the U.S.
  Fact 0.6: The public land was given away free to 1.6 million homesteaders.
  Fact 0.7: Most of the homesteads were west of the Mississippi River.
Sentence 1: An extension of the Homestead Principle in law , the Homestead Acts were an expression of the `` Free Soil '' policy of Northerners who wanted individual farmers to own and operate their own farms , as opposed to Southern slave - owners who wanted to buy up large tracts of land and use slave labor , thereby shutting out free white men .
  Fact 1.0: The Homestead Acts were an extension of the Homestead Principle in law.
  Fact 1.1: The Homestead Acts were an expression of the 'Free Soil' policy of Northerners.
  Fact 1.2: Northerners with the Free Soil policy wanted individual farmers to own and operate their own farms.
  Fact 1.3: Southern slave-owners, as opposed to Northerners, wanted to buy up large tracts of land.
  Fact 1.4: Southern slave-owners wanted to use slave labor on large tracts of land.
  Fact 1.5: Southern slave-owners' land purchases would shut out free white men.

Retrieved items:
<item 1>
Westward Expansion Trails
Westward expansion trails - wikipedia Westward expansion trails Jump to : navigation , search This article needs additional citations for verification . Please
help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( March 2013 ) ( Learn how and when to
remove this template message ) In the American Old West , overland trails were popular means of travel used by pioneers and immigrants throughout the 19th
century and especially between 1830 and 1870 as an alternative to sea and railroad transport . These immigrants began to settle various regions of North America
west of the Great Plains as part of the mass overland migrations of the mid-19th century . Settlers emigrating from the eastern United States were spurred by
various motives , among them religious persecution , economic incentives , and the prospect of new wealth and new freedoms in what was then a largely unspoiled
and unregulated country . By the 1840s , several popular and well - established routes had been blazed through the continental interior to destinations in the
far west , including the Oregon Trail , California Trail , and Mormon Trail .
</item 1>

<item 2>
Westward Expansion Trails
It was the only practical way for settlers in wagons without tools , livestock , or supplies to cross the mountains and usually thought critical to the
colonization of the American West . Some of the first to travel the Oregon Trail were Christian missionaries , members of the Methodist Episcopal Church who
established the Methodist Mission in 1834 . Even though they did n't make many converts , they were impressed by the short amount of time needed to reach the
Pacific Coast . Rumors about how the sun always shone there and wheat grew as tall as a man attracted American settlers . American settlers began following the
trail in 1841 , with the first recorded colonist wagon traingroup being the 1843 `` Great Migration '' of about 900 colonists , led by Marcus Whitman . The
Provisional Government of Oregon was established by such colonists in 1843 , generally limited to the Willamette Valley . The Oregon Treaty of 1846 divided the
Oregon Country between Great Britain and the United States , mostly along the 49th parallel . The Territory of Oregon was established shortly afterwards , in
1848 , and over 12,000 American settlers made the journey there during the decade .
</item 2>

<item 3>
Homestead Acts
The Free Soil Party of 1848 -- 52 , and the new Republican Party after 1854 , demanded that the new lands opening up in the west be made available to
independent farmers , rather than wealthy planters who would develop it with the use of slaves forcing the yeomen farmers onto marginal lands . Southern
Democrats had continually fought ( and defeated ) previous homestead law proposals , as they feared free land would attract European immigrants and poor
Southern whites to the west . After the South seceded and their delegates left Congress in 1861 , the Republicans and other supporters from the upper South
passed a homestead act . The intent of the first Homestead Act , passed in 1862 , was to liberalize the homesteading requirements of the Preemption Act of 1841
. Its leading advocates were Andrew Johnson , George Henry Evans and Horace Greeley . The homestead was an area of public land in the West ( usually 160 acres
or 65 ha ) granted to any US citizen willing to settle on and farm the land for at least five years .
</item 3>

<item 4>
Westward Expansion Trails
Unlike the more northern routes , pioneer wagons could travel year round , mountain passes not being blocked by snows , however it had the disadvantage of
summer heat and lack of water in the desert regions through which it passed in New Mexico Territory and the Colorado Desert of California . Subsequently , it
was a route of travel and commerce between the eastern United States and California . Many herds of cattle and sheep were driven along this route and it was
followed by the San Antonio - San Diego Mail Line in 1857 -- 1858 and then the Butterfield Overland Mail from 1858 -- 1861 . In 1846 , General Stephen Watts
Kearny and his dragoons with their scout Kit Carson found the route . One year later , Colonel Philip St. George Cooke and the Mormon Battalion followed Kearny
's route establishing the first wagon road . This wagon road became known as Cooke 's Wagon Road , or Sonora Road , until the discovery of gold brought a flood
of Americans westward in 1849 . The discovery of the Tucson Cutoff later that year shortened the route bringing it fully within the territory of the United
States .
</item 4>

<item 5>
Homestead Acts
Some scholars believe the acreage limits were reasonable when the act was written , but reveal that no one understood the physical conditions of the plains .
According to Hugh Nibley , much of the rain forest west of Portland , Oregon was acquired by the Oregon Lumber Company by illegal claims under the Act .
Nonetheless , in 1995 , a random survey of 178 members of the Economic History Association found that 70 percent of economists and 84 percent of economic
historians disagreed that `` Nineteenth - century U.S. land policy , which attempted to give away free land , probably represented a net drain on the productive
capacity of the country . '' Related Acts in other countries ( edit ) Canada ( edit ) Similar laws were passed in Canada : The Legislative Assembly of Ontario
passed The Free Grants and Homestead Act in 1868 , which introduced a conditional scheme to an existing free grant plan previously authorized by the Province of
Canada in The Public Lands Act of 1860 .
</item 5>

<item 6>
Westward Expansion Trails
Regardless of the trail used , the journey was often slow and arduous , fraught with risks from infectious diseases , dehydration , malnutrition , injury , and
harsh weather , with as many as one in ten travelers dying along the way , usually as a result of disease . The history of these trails and the pioneers who
traveled them have since become deeply embedded in the culture and folklore of the United States as some of the most significant influences to shape the content
and character of the nation . The remains of many trail ruts can still be observed in various locations throughout the American West . Travelers may loosely
follow various routes of the emigrant trails on modern highways through the use of byway signs across the western states . Contents ( hide ) 1 Trail choices 2
Santa Fe Trail 3 Old Spanish Trail 4 Oregon Trail 5 California Trail 6 Mormon Trail 7 Southern Emigrant Trail 8 See also 9 References Trail choices ( edit )
Travelers across what became the Western United States in the 19th century had the choice of several routes . Some of the earliest were those of the Mexicans in
the southwest . American trade with Northern Mexico created the Santa Fe Trail between St.
</item 6>

<item 7>
Westward Expansion Trails
During the twenty - five years 1841 -- 1866 , 250,000 to 650,000 people `` pulled up stakes , '' and headed west along these trails . About one - third
immigrated to Oregon , one - third to California and one - third to Utah , Colorado , and Montana . Although it is often stated that the Northern trails began
in certain cities on the Missouri River , emigrants following any of the three trails typically left from one of three `` jumping off '' points on the Missouri
's steamboat serviced river ports : Independence , Missouri or Saint Joseph , Missouri , or Council Bluffs , Iowa . ( Once known as Kanesville , Iowa until 1852
; after river dredging in the early 1850s , the latter town at the Missouri - Platte confluence became the most common departure point since it was close in
proximity to the River Platte -- along which the eastern trails ascend to South Pass above Fort Laramie. ) The trails from these cities ( and several others )
converged in the mostly empty flatlands of central Nebraska near present - day Kearney , in the vicinity of Fort Kearney .
</item 7>

<item 8>
Westward Expansion Trails
Up to 50,000 people , or one - tenth of the emigrants who attempted the crossing continent , died during the trip , most from infectious disease such as cholera
, spread by poor sanitation : with thousands traveling along or near the same watercourses each summer , downstream travelers were susceptible to ingesting
upstream wastewater including bodily waste . Hostile confrontations with Native Americans , although often feared by the emigrants , were comparatively rare ,
prior to the American Civil War . Most emigrants traveled in large parties or `` trains '' of up to several hundred wagons led by a wagon master . In 1859 the
government published a guidebook written by Captain Randolph B. Marcy , called The Prairie Traveler , in order to help emigrants prepare for the journey. >
Santa Fe Trail ( edit ) Main article : Santa Fe Trail The Santa Fe Trail was a 19th - century transportation route through central North America that connected
Independence , Missouri with Santa Fe , New Mexico . Pioneered in 1821 by William Becknell , it served as a vital commercial highway until the introduction of
the railroad to Santa Fe in 1880 .
</item 8>

<item 9>
Westward Expansion Trails
After the end of the Mexican -- American War in 1849 , vast new American conquests again enticed mass immigration . Legislation like the Donation Land Claim Act
and significant events like the California Gold Rush further lured colonists to travel overland to the west . Two major wagon - based transportation networks ,
one typically starting in Missouri and the other in the Mexican province of Santa Fe de Nuevo México , served the majority of migrants during the era of
westward expansion . Three of the Missouri - based routes -- the Oregon , Mormon , and California Trails -- were collectively known as the Emigrant Trails .
Historians have estimated at least 500,000 emigrants used these three trails between 1843 and 1869 , and despite growing competition from transcontinental
railroads , some use even continued into the early 20th century . The major southern routes were the Santa Fe Trail , the Southern Emigrant Trail , and the Old
Spanish Trail , as well as its wagon road successor the Mormon Road , a southern spur of the California Trail used in the winter that also made use of the
western half of the Old Spanish Trail .
</item 9>

<item 10>
Homestead Acts
History ( edit ) Donation land claim Act of 1850 ( edit ) Main article : Donation Land Claim Act The Donation Land Claim Act allowed settlers to claim land in
the Oregon Territory , then including the modern states of Washington , Oregon , Idaho and parts of Wyoming . Settlers were able to claim 320 or 640 acres of
land for free between 1850 and 1854 , and then at a cost of $1.25 per acres until the law expired in 1855 . Homestead Act of 1862 ( edit ) Certificate of
homestead in Nebraska given under the Homestead Act , 1862 Wikisource has original text related to this article : Homestead Act ( 1862 ) The `` yeoman farmer ''
ideal of Jeffersonian democracy was still a powerful influence in American politics during the 1840 -- 1850s , with many politicians believing a homestead act
would help increase the number of `` virtuous yeomen '' .
</item 10>
=== END ITEM id=8a84eac959c868f8 ===

=== ITEM id=736171a548e341af ===
Gold sentences and facts:
Sentence 0: On 19 February 2007 Disney lost a court case in Los Angeles which ruled their `` misguided claims '' to dispute the licensing agreements with Slesinger , Inc. were unjustified , but a federal ruling of 28 September 2009 , again from Judge Florence - Marie Cooper , determined that the Slesinger family had granted all trademark and copyright rights to Disney , although Disney must pay royalties for all future use of the characters . Both parties have expressed satisfaction with the outcome .
  Fact 0.0: On 19 February 2007, Disney lost a court case in Los Angeles.
  Fact 0.1: That court ruled that Disney's misguided claims to dispute the licensing agreements with Slesinger, Inc. were unjustified.
  Fact 0.2: A federal ruling of 28 September 2009 was again from Judge Florence-Marie Cooper.
  Fact 0.3: The federal ruling of 28 September 2009 determined that the Slesinger family had granted all trademark and copyright rights to Disney.
  Fact 0.4: Disney must pay royalties for all future use of the characters.
  Fact 0.5: Both parties have expressed satisfaction with the outcome.

Retrieved items:
<item 1>
Winnie-the-Pooh
A further special story , Winnie - the - Pooh Meets the Queen , was published in 2016 to mark the 90th anniversary of Milne 's creation and the 90th birthday of
Elizabeth II . It sees Winnie the Pooh meet the Queen at Buckingham Palace . Stephen Slesinger On 6 January 1930 , Stephen Slesinger purchased US and Canadian
merchandising , television , recording and other trade rights to the `` Winnie - the - Pooh '' works from Milne for a $1000 advance and 66 % of Slesinger 's
income , creating the modern licensing industry . By November 1931 , Pooh was a $50 million - a-year business . Slesinger marketed Pooh and his friends for more
than 30 years , creating the first Pooh doll , record , board game , puzzle , US radio broadcast ( NBC ) , animation , and motion picture film . Red shirt Pooh
The first time Pooh and his friends appeared in colour was 1932 , when he was drawn by Slesinger in his now - familiar red shirt and featured on an RCA Victor
picture record .
</item 1>

<item 2>
“Winnie-the-Pooh Meets the Queen,” published in 2016, commemorated both Pooh’s 90th anniversary and Queen Elizabeth II’s 90th birthday. In 1930, Stephen
Slesinger acquired US and Canadian rights to Pooh for a $1,000 advance and 66% of his income. By 1931, Pooh was a $50-million-a-year business. Over three
decades, Slesinger created many Pooh products and media
</item 2>

<item 3>
Disney has produced numerous Pooh adaptations, including TV and direct-to-video works and four theatrical films. Pooh merchandise generates substantial revenue,
prompting a long dispute: Slesinger, Inc. alleged underreported sales and unpaid royalties. After years of litigation and appeals, courts upheld Disney’s
rights, while requiring royalties for future use; both sides accepted the 2009 ruling. Stage adaptations included plays and musicals from 1931 onward, including
a 1977 Christmas production.
</item 3>

<item 4>
Winnie-the-Pooh
These have included theatrical featurettes , television series , and direct - to - video films , as well as the theatrical feature - length films The Tigger
Movie , Piglet 's Big Movie , Pooh 's Heffalump Movie , and Winnie the Pooh . Merchandising revenue dispute Pooh videos , soft toys , and other merchandise
generate substantial annual revenues for Disney . The size of Pooh stuffed toys ranges from Beanie and miniature to human - sized . In addition to the stylised
Disney Pooh , Disney markets Classic Pooh merchandise which more closely resembles E.H. Shepard 's illustrations . In 1991 , Stephen Slesinger , Inc. filed a
lawsuit against Disney which alleged that Disney had breached their 1983 agreement by again failing to accurately report revenue from Winnie the Pooh sales .
Under this agreement , Disney was to retain approximately 98 % of gross worldwide revenues while the remaining 2 % was to be paid to Slesinger . In addition ,
the suit alleged that Disney had failed to pay required royalties on all commercial exploitation of the product name .
</item 4>

<item 5>
Winnie-the-Pooh
Jump up ^ `` The Curse of Pooh . '' Fortune . Jump up ^ `` The Pooh Files '' The Albion Monitor . Jump up ^ Nelson , Valerie J ( 20 July 2007 ) . `` Shirley
Slesinger Lasswell , 84 ; fought Disney over Pooh royalties '' . Los Angeles Times . Retrieved 14 August 2007 . Jump up ^ `` Judge dismisses Winnie the Pooh
lawsuit '' The Disney Corner . Jump up ^ James , Meg ( 26 September 2007 ) . `` Disney wins lawsuit ruling on Pooh rights '' . Los Angeles Times . Retrieved 26
September 2007 . Jump up ^ `` Winnie the Pooh goes to court '' USA Today Jump up ^ `` USATODAY.com -- Justices wo n't hear copyright appeal by relative of
Winnie the Pooh '' . USA Today . Associated Press . Jump up ^ `` Disney loses court battle in Winnie the Pooh copyright case '' . ABC News . 17 February 2007 .
Retrieved 15 May 2008 .
</item 5>

<item 6>
Winnie-the-Pooh
were unjustified , but a federal ruling of 28 September 2009 , again from Judge Florence - Marie Cooper , determined that the Slesinger family had granted all
trademark and copyright rights to Disney , although Disney must pay royalties for all future use of the characters . Both parties have expressed satisfaction
with the outcome . Adaptations Theatre Winnie - the - Pooh at the Guild Theater Sue Hastings Marionettes , 1931 Winnie - the - Pooh , a play in three acts
Dramatized by Kristin Sergel , Dramatic Publishing Company , 1957 Winnie - the - Pooh , a musical comedy in two acts Lyrics by A.A. Milne and Kristin Sergel ,
Music by Allan Jay Friedman , Book by Kristin Sergel , Dramatic Publishing Company , 1964 A Winnie - the - Pooh Christmas Tail , In Which Winnie - the - Pooh
and His Friends Help Eeyore Have a Very Merry Christmas ( Or a Very Happy Birthday . ) Book , Music and Lyrics by James W. Rogers , Dramatic Publishing Company
, 1977 `` Bother !
</item 6>

<item 7>
Winnie-the-Pooh
17 February 2007 . Retrieved 15 May 2008 . Jump up ^ James , Meg ( 29 September 2009 ) . `` Pooh rights belong to Disney , judge rules '' . Los Angeles Times .
Retrieved 5 October 2009 . Jump up ^ Shea , Joe ( 4 October 2009 ) . `` The gordian knot of Pooh rights is finally untied in federal court '' . The American
Reporter . Retrieved 5 October 2009 . Jump up ^ `` Hastings Marionettes : Will Open Holiday Season at Guild Theatre on Saturday '' . The New York Times . 22
December 1931 . p. 28 . Jump up ^ `` Ian Carmichael And Full Cast -- The House at Pooh Corner -- HMV Junior Record Club -- UK -- 7EG 117 '' . 45cat. 23 July
2010 . Retrieved 5 November 2011 . Jump up ^ `` Winnie the Pooh '' . OCLC WorldCat . Retrieved 15 January 2016 .
</item 7>

<item 8>
The text is a collection of references about Winnie-the-Pooh, covering a 2009 sequel and a 2016 story featuring a penguin and the Queen; Pooh’s commercial and
literary history; books, recordings, animation, and Disney live-action films; Russian adaptations and Willie Rushton; philosophical and cultural commentary; and
Pooh’s status as an English icon, top-earning fictional character, and Hollywood Walk of Fame honoree. It also cites Oxford University Press, World
</item 8>

<item 9>
The citations document disputes over Winnie-the-Pooh royalties and copyright, including lawsuits involving Disney and the Slesinger family. Reports from
2007–2009 describe court rulings, appeals, and a 2009 federal decision affirming Disney’s rights. Other references cover a 1931 marionette performance, a UK
audio recording, and a WorldCat catalog entry.
</item 9>

<item 10>
Winnie-the-Pooh, created by A.A. Milne and illustrated by E.H. Shepard, evolved from Christopher Robin’s teddy bear, first called Edward in 1924; the character
debuted as Pooh in 1925. Inspired by Ashdown Forest, the stories became classic books, poems, recordings, stage works, and Disney films and television series.
Pooh’s popularity fueled extensive merchandising and a long royalties dispute, resolved largely in Disney’s favor in 200
</item 10>
=== END ITEM id=736171a548e341af ===
