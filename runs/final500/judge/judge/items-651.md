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

=== ITEM id=bbd9861fb80e96d6 ===
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
studios ) 3.2 Notable guests and events 4 Distribution 5 See also 6 References and footnotes 7 External links Format ( edit ) The program 's format consists of
an extended one - on - one interview with the host , and is sometimes followed by a roundtable discussion or one - on - two interview with figures in
adversarial positions , either Congressional members from opposite sides of the aisle or political commentators . Originally a half - hour program for most of
its history , the show expanded to 60 minutes starting with the September 20 , 1992 , broadcast . The program also features in - depth examinations of facts
behind political and general news stories ( particularly as part of a segment called the `` Data Download '' , introduced after Chuck Todd assumed duties as
moderator , which is conducted on a touchscreen within the main set . History ( edit ) Meet the Press set , November 1975 . On this broadcast , a sitting
American president ( Gerald Ford ) was , for the first time , a guest on a live television network news program .
</item 1>

<item 2>
Meet the Press
On May 2 , 2010 , Meet the Press became the last NBC News program to convert to high definition , and unveiled a new set consisting of large video screens
mostly used to display Washington scenery , satellite interview subjects and moderator and subject talking points , along with graphics produced for the format
. Moderators ( edit ) The following is the list of moderators for Meet the Press : Martha Rountree 1947 -- 1953 Ned Brooks 1953 -- 1965 Lawrence E. Spivak 1966
-- 1975 Bill Monroe 1975 -- 1984 Roger Mudd and Marvin Kalb ( co-moderators ) 1984 -- 1985 Marvin Kalb 1985 -- 1987 Chris Wallace 1987 -- 1988 Garrick Utley
1989 -- 1991 Tim Russert 1991 -- 2008 Tom Brokaw 2008 David Gregory 2008 -- 2014 Chuck Todd 2014 -- present On - remote broadcasts ( outside D.C. studios ) (
edit ) Date Location Event / Reason July 17 , 1988 Atlanta , Georgia 1988 Democratic National Convention August 14 , 1988 New Orleans ,
</item 2>

<item 3>
Meet the Press
The following week on August 17 , he moderated the entire broadcast . It was also reported on December 1 , 2008 , that the December 7 broadcast would be Brokaw
's last , with Gregory becoming the new permanent host the following Sunday . Under Gregory ( edit ) David Gregory began his tenure as moderator on December 14
, 2008 . Four days after Gregory 's first regular broadcast , on December 18 , NBC News political director Chuck Todd was named contributing editor of Meet the
Press . Throughout Gregory 's tenure as moderator , Meet the Press experienced significant ratings declines . In the final three months of 2013 , the program
placed third among the Sunday morning talk shows in total viewership , behind CBS 's Face the Nation and ABC 's This Week for the first time since 1992 ; it
also experienced the lowest ratings in the show 's entire history among the key 25 - to - 54 age viewing demographic during this period . NBC management became
uncertain as to the future direction of the program .
</item 3>

<item 4>
Meet the Press
This section , in the next two paragraphs , needs additional citations for verification . Please help improve this article by adding citations to reliable
sources . Unsourced material may be challenged and removed . ( September 2014 ) ( Learn how and when to remove this template message ) Meet the Press was
originally presented as a 30 - minute press conference with a single guest and a panel of questioners . Its first guest was James Farley , who served as
Postmaster General , Democratic National Committee chairman and campaign manager to Franklin Delano Roosevelt under the first two terms of the New Deal
Administration . Creator Rountree served as its first host , the program 's only female moderator to date . She stepped down on November 1 , 1953 , and was
succeeded by Ned Brooks , who remained as moderator until his retirement on December 26 , 1965 . Spivak became the moderator on January 1 , 1966 , moving up
from his role as a permanent panelist . He retired on November 9 , 1975 , on a special one - hour edition that featured , for the first time , a sitting
president , Gerald Ford , as guest .
</item 4>

<item 5>
Meet the Press
Meet the Press - Wikipedia Meet the Press Jump to : navigation , search This article is about the American public affairs program . For the Australian program ,
see Meet the Press ( Australian TV program ) . Meet the Press Genre Public affairs / news analysis program Created by Martha Rountree Lawrence E. Spivak
Directed by Rob Melick Presented by Chuck Todd ( for past moderators , see section ) Narrated by Fred Facey Bert Pence Dennis Haysbert Theme music composer John
Williams Opening theme `` The Pulse of Events '' ( fourth part of The Mission ) Country of origin United States Original language ( s ) English No. of seasons
70 No. of episodes 3,600 + ( weekly ) ( as of 2017 ) Production Executive producer ( s ) John Reiss Location ( s ) NBC News Washington Bureau , Washington ,
D.C.
</item 5>

<item 6>
Meet the Press
Russert interviews General Peter Pace in 2006 . Network officials , concerned for the show 's future , turned to Tim Russert , the network 's Washington , D.C.
, bureau chief . He took over as moderator of Meet the Press on December 8 , 1991 , and remained with the program until his death on June 13 , 2008 , becoming
the longest serving moderator in the program 's history . Under Russert , the program was expanded to one hour and became less of a televised press conference ,
focusing more on Russert 's questions and comments ; Russert also engaged in longer in - depth interviews and hosted panels of experts to discuss the topics
featured in that week 's broadcast . Russert signed off each edition by saying , `` That 's all for today . We 'll be back next week . If it 's Sunday , it 's
Meet the Press . '' During the professional football season , Russert , a native of Buffalo , New York , and an avid fan of the Buffalo Bills , sometimes added
, `` Go Bills ! , '' and occasionally would ask panelists , `` How ' bout those Sabres ? ''
</item 6>

<item 7>
Meet the Press
It originates from NBC 's bureau in Washington , D.C. The longevity of Meet the Press can be illustrated in consideration of the fact that the program debuted
during what was only the second official `` network television season '' for American television . One historical landmark of the program is that it was the
first on which a sitting U.S. President , Gerald Ford , appeared on a live television network news program , which occurred on the November 9 , 1975 broadcast .
The program has been hosted by 12 different moderators to date , beginning with creator Martha Rountree . The show 's moderator since 2014 is Chuck Todd , who
also serves as political director for NBC News . Currently , the hour - long program airs in most markets on Sundays at 9 : 00 a.m. live in the Eastern Time
Zone and on tape delay elsewhere . Meet the Press is also occasionally pre-empted due to network coverage of sports events held outside of the U.S. The program
is also rebroadcast on Sundays at 2 : 00 p.m. , and Mondays at 2 : 00 a.m. and sometimes 4 : 00 a.m.
</item 7>

<item 8>
Meet the Press
The next week , Bill Monroe , previously a weekly panelist like Spivak had been years before , took over as moderator and stayed until June 2 , 1984 . For the
next seven and a half years , the program then went through a series of hosts as it struggled in the ratings against ABC 's This Week with David Brinkley .
Roger Mudd and Marvin Kalb , as co-moderators , followed Monroe for a year , followed by Chris Wallace ( who would later to go on to a much longer run as host
of the rival program Fox News Sunday ) from 1987 to 1988 . Garrick Utley , then hosting Weekend Today , concurrently hosted Meet the Press from 1989 through
December 1 , 1991 . All this occurred despite the increasing ratings of NBC News ' other programs ( and those of the network generally ) during that period .
The program originally aired at noon Eastern Time every Sunday , before moving to a 9 : 00 a.m. slot by the early 1990s . Under Russert ( edit ) Meet the Press
logo used from September 10 , 1995 to June 2008 .
</item 8>

<item 9>
Meet the Press
Camera setup Multi-camera Running time 30 minutes ( 1947 -- 1992 ) 52 minutes ( 1992 -- present ) Production company ( s ) NBC News Productions Release Original
network NBC Picture format 480i ( 4 : 3 SDTV ) ( 1947 -- 2010 ) 720p ( 4 : 3 SDTV ) ( 1996 -- 2010 ) 1080i ( 16 : 9 HDTV ) ( 2010 -- present ) Original release
November 6 , 1947 ( 1947 - 11 - 06 ) -- present External links Website www.nbcnews.com/meet-the-press Meet the Press is a weekly American television news /
interview program broadcast on NBC . It is the longest - running program in television history , though the current format bears little resemblance to the debut
episode on November 6 , 1947 . Meet the Press specializes in interviews with leaders in Washington , D.C. , across the country and even the world on issues of
politics , economics , foreign policy and other public affairs , along with panel discussions that provide opinions and analysis . It originates from NBC 's
bureau in Washington , D.C.
</item 9>

<item 10>
Meet the Press
Bureau Chief and was given `` executive oversight '' of Meet the Press . Interim Brokaw era ( edit ) NBC Nightly News anchor Brian Williams acted as moderator
of the first show following the June 15 tribute to Russert , with the same guests and subject matter that Russert was planning for when he died . Following
Russert 's death , Tom Brokaw was named the interim moderator through the 2008 general elections . Brokaw followed Russert 's tradition by signing off with ``
We 'll be back next Sunday because if it 's Sunday , it 's Meet the Press '' ( a sign - off that would continue to be used by his successors as moderator ) . In
September of that year , the show was presented with limited commercial interruption . On August 10 , David Gregory moderated the panel discussion during the
second half - hour of the broadcast , while Brokaw anchored the first half - hour from the site of the Summer Olympics in Beijing . The following week on August
17 , he moderated the entire broadcast .
</item 10>
=== END ITEM id=bbd9861fb80e96d6 ===

=== ITEM id=88143e35d14775c8 ===
Gold sentences and facts:
Sentence 0: Health or vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually rewarded in some way .
  Fact 0.0: Health or vitality is an attribute assigned to entities in a role-playing or video game.
  Fact 0.1: Health or vitality is assigned to entities such as the player character, enemies, and objects.
  Fact 0.2: Health or vitality indicates the state of an entity in combat.
  Fact 0.3: Health is usually measured in hit points or health points.
  Fact 0.4: Hit points or health points are shortened to HP.
  Fact 0.5: When the HP of a player character reaches zero, the player may lose a life.
  Fact 0.6: When the HP of a player character reaches zero, the character might become incapacitated or die.
  Fact 0.7: When the HP of an enemy reaches zero, the enemy may be defeated or die.
  Fact 0.8: When the HP of an enemy reaches zero, the player is usually rewarded in some way.

Retrieved items:
<item 1>
Harry Potter (film series)
Harry is also beset by disturbing and realistic nightmares , while Professor Umbridge , a representative of Minister for Magic Cornelius Fudge , is the new
Defence Against the Dark Arts teacher . Harry becomes aware that Voldemort is after a prophecy which reveals : `` neither can live while the other survives '' .
The rebellion involving the students of Hogwarts , secret organisation Order of the Phoenix , the Ministry of Magic , and the Death Eaters begins . Harry Potter
and the Half - Blood Prince ( 2009 ) Main article : Harry Potter and the Half - Blood Prince ( film ) In Harry 's sixth year at Hogwarts , Lord Voldemort and
his Death Eaters are increasing their terror upon the Wizarding and Muggle worlds . Headmaster Albus Dumbledore persuades his old friend Horace Slughorn to
return to Hogwarts as a professor as there is a vacancy to fill . There is a more important reason , however , for Slughorn 's return . While in a Potions
lesson , Harry takes possession of a strangely annotated school textbook , inscribed ' This is the property of the Half - Blood Prince ' . Draco Malfoy
struggles to carry out a deed presented to him by Voldemort .
</item 1>

<item 2>
Harry Potter (film series)
2001 ) The Chamber of Secrets ( 2002 ) The Prisoner of Azkaban ( 2004 ) The Goblet of Fire ( 2005 ) The Order of the Phoenix ( 2007 ) The Half - Blood Prince (
2009 ) The Deathly Hallows -- Part 1 ( 2010 ) The Deathly Hallows -- Part 2 ( 2011 ) Music The Philosopher 's Stone The Chamber of Secrets The Prisoner of
Azkaban The Goblet of Fire The Order of the Phoenix The Half - Blood Prince The Deathly Hallows -- Part 1 The Deathly Hallows -- Part 2 Related Cast members
Production of The Deathly Hallows Characters Main Harry Potter Ron Weasley Hermione Granger Lord Voldemort Albus Dumbledore Severus Snape Rubeus Hagrid Draco
Malfoy Supporting Ginny Weasley Fred and George Weasley Hogwarts staff Order of the Phoenix Dumbledore 's Army Death Eaters Fictional universe Hogwarts Magic
Magical creatures Magical objects Ministry of Magic Muggle Places Potions Quidditch Related works Fantastic Beasts and Where to Find Them Quidditch Through the
Ages The Tales of Beedle the Bard Prequel Pottermore The Cursed Child Hogwarts
</item 2>

<item 3>
Health (gaming)
US Navy used similar concept in their tactical war games already in 1920s and 1930s . In their simulation each ship had Life parameter . The unit of Life of the
ship was a number of `` equivalent penetrative 14 - inch shell hits '' . The Navy considered ie. that Kongō - class battlecruiser had 12 Life points and Nagato
- class battleship had 18.8 . A visual power meter representing stamina was used in Nintendo 's 1983 arcade game Punch - Out ! ! and Data East 's 1981 DECO
Cassette System arcade game Flash Boy . Usage ( edit ) In action video games as well as in role - playing games , health points can usually be depleted by
attacking the entity . A defense attribute might reduce the amount of HP that is lost when a character is damaged . It is common in role - playing games for a
character 's maximum health and defense attributes to be gradually raised as the character levels up . In game design , it is deemed important that a player is
aware of it when they are losing health , each hit playing a clear sound effect .
</item 3>

<item 4>
Harry Potter (film series)
Draco Malfoy struggles to carry out a deed presented to him by Voldemort . Meanwhile , Dumbledore and Harry secretly work together to discover the method on how
to destroy the Dark Lord once and for all . Harry Potter and the Deathly Hallows -- Part 1 ( 2010 ) Main article : Harry Potter and the Deathly Hallows -- Part
1 The Potters ' Cottage from the final film . After unexpected events at the end of the previous year , Harry , Ron , and Hermione are entrusted with a quest to
find and destroy Lord Voldemort 's secret to immortality -- the Horcruxes . It is supposed to be their final year at Hogwarts , but the collapse of the Ministry
of Magic and Voldemort 's rise to power prevents them from attending . The trio undergo a long journey with many obstacles in their path including Death Eaters
, Snatchers , the mysterious Deathly Hallows , and Harry 's connection with the Dark Lord 's mind becoming ever stronger .
</item 4>

<item 5>
Health (gaming)
Health ( Gaming ) - wikipedia Health ( Gaming ) Jump to : navigation , search A health bar , a possible representation of the health of a character . Health or
vitality is an attribute assigned to entities , such as the player character , enemies and objects within a role - playing or video game , that indicates its
state in combat . Health is usually measured in hit points or health points , shortened to HP . When the HP of a player character reaches zero , the player may
lose a life or their character might become incapacitated or die . When the HP of an enemy reaches zero , it may be defeated or die and the player is usually
rewarded in some way . Any entity within a game could have a health value , including the player character , non-player characters and objects . Indestructible
entities have no diminishable health value . Health might be displayed as a numeric value , such as `` 50 / 100 '' . Here , the first number indicates the
current amount of HP an entity has and the second number indicates the entity 's maximum HP .
</item 5>

<item 6>
Health (gaming)
This mechanic initially appeared in action role - playing games , with early examples including the Hydlide series , the Ys series , and Woody Poco . In Woody
Poco , the rate at which health recharges is based on food level . In Hydlide and Ys , the player character has to stand still for their health to automatically
regenerate . This system was popularized in first - person shooters by Halo : Combat Evolved ( 2001 ) , though regenerating health in The Getaway ( 2002 ) has
been cited to be more comparable to later use of the mechanic in first - person shooters . Display ( edit ) Heart - shaped icons can indicate the amount of
health a player has left . The way health is displayed on the screen has an effect on the player . Many games only show the health of the player character ,
while keeping the health of enemies hidden . This is done in the Legend of Zelda series , Minecraft and Monster Hunter series to keep the player 's progress in
defeating their enemy unclear and therefore exciting . In these games , the fact that the enemies are being damaged is indicated by their behavior .
</item 6>

<item 7>
Health (gaming)
Author Scott Rogers states that `` health should deplete in an obvious manner , because with every hit , a player is closer to losing their life . '' The
display of health also helps to dramatize the near - loss of a life . Regeneration ( edit ) Player characters can often restore their health points by consuming
certain items , such as health potions , food or first - aid kits . Staying a night at an inn fully restores a character 's health in many role - playing video
games . In general , the different methods of regenerating health has its uses in a particular genre . In action games , this method is very quick , whereas
role - playing games feature slower paced methods to match the gameplay and realism . Some video games feature automatically regenerating health , where lost
health points are regained over time . This can be useful to not `` cripple '' the player , making them still able to continue even after losing lots of health
. However , automatically regenerating health may also cause a player to `` power through '' sections they might otherwise have had to approach cautiously ,
simply because there are no lasting consequences to losing a large amount of health .
</item 7>

<item 8>
Health (gaming)
In these games , the fact that the enemies are being damaged is indicated by their behavior . On the other hand , many fighting games , such as the Street
Fighter series , use easy - to - read health bars to clearly indicate the progress the player is making with each hit . It is common in first - person shooters
to indicate low health of the player character by blood spatters or by a distorted red hue on the screen , attempting to mimic the effects of wounding and
trauma . These visual effects fade as health regenerates . References ( edit ) ^ Jump up to : Moore , Michael ( 2011 - 03 - 23 ) . Basics of Game Design . CRC
Press . pp. 151 , 194 . ISBN 1439867763 . Retrieved 2014 - 12 - 09 . Jump up ^ Chris Antista . `` The 10 most creative life bars '' . GamesRadar. 2010 - 08 - 17
. p. 2 . Archived from the original on 2014 - 12 - 28 .
</item 8>

<item 9>
Health (gaming)
In video games , health can also be displayed graphically , such as with a bar that empties itself when an entity loses health ( a health bar , typically red )
, icons that are `` chipped away '' , or in more novel ways . Contents ( hide ) 1 History 2 Usage 2.1 Regeneration 3 Display 4 References History ( edit )
Dungeons & Dragons co-creator Dave Arneson described the origin of hit points in a 2002 interview . When Arneson was adapting the medieval wargame Chainmail (
1971 ) to a fantasy setting , a process that with Gary Gygax would lead to Dungeons & Dragons , he saw that the emphasis of the gameplay was moving from large
armies to small groups of heroes and eventually to the identification of one player and one character that is essential to role - playing as it was originally
conceived . Players became attached to their heroes and did not want them to die every time they lost a die roll . Players were thus given multiple hit points
which were incrementally decreased as they took damage . Arneson took the concept , along with armor class , from a set of a naval American Civil War game 's
rules .
</item 9>

<item 10>
Harry Potter (film series)
In 2006 , Nicholas Hooper started work on the score to Order of the Phoenix by reuniting with director David Yates . Hooper also composed the soundtrack to Half
- Blood Prince but decided not to return for the final films . In January 2010 , Alexandre Desplat was confirmed to compose the score for Harry Potter and the
Deathly Hallows -- Part 1 . The film 's orchestration started in the summer with Conrad Pope , the orchestrator on the first three Harry Potter films ,
collaborating with Desplat . Pope commented that the music `` reminds one of the old days . '' Desplat returned to score Harry Potter and the Deathly Hallows --
Part 2 in 2011 . Director David Yates stated that he wanted John Williams to return to the series for the final installment , but their schedules did not align
due to the urgent demand for a rough cut of the film . The final recording sessions of Harry Potter took place on 27 May 2011 at Abbey Road Studios with the
London Symphony Orchestra , orchestrator Conrad Pope , and composer Alexandre Desplat .
</item 10>
=== END ITEM id=88143e35d14775c8 ===

=== ITEM id=5cbcc0f8fd70c458 ===
Gold sentences and facts:
Sentence 0: Sports - related costs for the Summer Games since 1960 is on average USD 5.2 billion and for the Winter Games USD 3.1 billion dollars . Severe cost overruns made the 2014 Winter Olympics the most expensive Olympics in history ; with Russian politician Boris Nemtsov citing allegations of corruption among government officials , and Allison Stewart of the Saïd Business School at Oxford citing tight relationships between the government and construction firms . While originally budgeted at US $12 billion , the budget expanded to US $51 billion , surpassing the estimated $44 billion cost of the 2008 Summer Olympics in Beijing , making Sochi the most expensive Olympics in history .
  Fact 0.0: Sports-related costs for the Summer Games since 1960 are on average USD 5.2 billion.
  Fact 0.1: Sports-related costs for the Winter Games are on average USD 3.1 billion.
Sentence 1: Host City Year Final Operating Budget Total Costs Taxpayer Contribution Profit / Loss Year Debt Paid Off Notes Athens Summer Olympics 1896 3,740,000 ₯ Donations by George Averoff of 1,000,000 ₯ covered potential losses London Summer Olympics 1908 US $394,000 est . £ 6,377 Chamonix Winter Olympics 1924 3,500,000 ₣ 2,000,000 ₣ Paris Summer Olympics 1924 5,496,610 ₣ Amsterdam Summer Olympics 1928 $1,183,000 $18,000 St. Moritz Winter Olympics 1928 CHF 706,000 CHF104 , 800 Los Angeles Summer Olympics 1932 $1,000,000 Berlin Summer Olympics 1936 $30,000,000 London Summer Olympics 1948 £ 761,688 £ 29,000 Helsinki Summer Olympics 1952 1,580,000,000 mk 49,000,000 mk Melbourne Summer Olympics 1956 A £ 5,400,000 A £ 300,000 Tokyo Summer Olympics 1964 US $ 72,000,000 Mexico City Summer Olympics 1968 US $ 176,000,000 Munich Summer Olympics 1972 1,972,000,000 DM Montreal Summer Olympics 1976 CDN $ 207,000,000 CDN $ 1,410,000,000 CDN $ 990,000,000 2006 A special tobacco tax was introduced in May 1976 to fund the loss Lake Placid Winter Olympics 1980 US $ 49,000,000 US $ 169,000,000 Moscow Summer Olympics 1980 US $ 231,000,000 US $ 1,350,000,000 USD $1,190,000,000 deficit This was the year that The United States and 64 other Nations boycotted due to the Soviet invasion of Afghanistan . Sarajevo Winter Olympics 1984 US $ 55,400,000 US $ 110,900,000 US $10,000,000 The first Olympics since 1948 to make a profit Los Angeles Summer Olympics 1984 US $ 320,000,000 US $ 413,000,000 $75,000,000 US $250,000,000 1984 The first Summer Olympics since 1932 to make a profit Calgary Winter Olympics 1988 CDN $ 438,000,000 CDN $ 899,000,000 CDN $ 425,000,000 CDN $ 32,000,000 1988 Seoul Summer Olympics 1988 USD $4,000,000,000 US $300,000,000 1988 A record profit for a government - run Olympiad Barcelona Summer Olympics 1992 US $ 850,000,000 US $ 9,300,000,000 US $10,000,000 1992 Operating costs were put at 9.1 % of the total cost . The vast majority of spending was to improve infrastructure . Albertville Winter Olympics 1992 USD $1,200,000,000 on infrastructure USD $67,000,000 Lillehammer Winter Olympics US $ 1,100,000,000 US $ 250,000,000 Atlanta Summer Olympics US $ 1,800,000,000 US $ 609,000,000 US $ 19,000,000 Despite the profit , Atlanta 's heavy reliance on corporate sponsorship drew criticism of commercialization Nagano Winter Olympics 1998 ~ US $ 10,000,000,000 in new infrastructure Net loss Estimated 2015 The full cost of the Nagano Olympics is unknown , due to Nagano Olympic Bid Committee vice-secretary general Sumikazu Yamaguchi ordering accounting documents burned Sydney Summer Olympics 2000 A $ 6,600,000,000 A $ 3,000,000,000 ( A $ 363 , 5000 , 000 borne by the public ) A $ 2,050,000,000 USD $2,100,000,000 loss 2000 Salt Lake City Winter Olympics 2002 US $ 2,000,000,000 US $ 1,200,000,000 US $ 600,000,000 US $ 101,000,000 2002 Additional security costs were incurred in the wake of the September 11 attacks Athens Summer Olympics US $ 15,000,000,000 US $ 9,000,000,000 US $ 6,200,000,000 US $ 14,500,000,000 The cost of the 2004 Athens Summer Games has been cited as a contributor to the Greek government - debt crisis . Many of the venues lie vacant and rotting ; the Independent newspaper reports as many as 21 out of 22 are unused . Turin Winter Olympics 2006 US $ 700,000,000 USD $3,200,000 The Italian government created a lottery game to cover its financial losses . Beijing Summer Olympics 2008 US $ 44,000,000,000 CNY 1,000,000,000 ( US $ 146,000,000 ) 2008 Vancouver Winter Olympics CDN $ 1,700,000,000 ( US $ 1,260,000,000 ) US $ 6,400,000,000 US $ 2,300,000,000 CDN $ 1,900,000 Included in the total US $ 6,400,000,000 cost are the $1,000,000,000 for security , $2,500,000,000 for transportation extensions and upgrades , and $900,000,000 for the new Vancouver Convention Centre ( An additional $554,000,000 was spent by the city including a portion on the Olympic Village ) . London Summer Olympics 2012 US $ 10,400,000,000 US $ 14,600,000,000 US $ 4,400,000,000 GBP £ nil 2012 Additional costs include $90 million for converting the Olympic Stadium ( London ) to a football venue Sochi Winter Olympics 2014 US $ 51,000,000,000 US $ 53,150,000 The most expensive Olympic Games in history , surpassing the previous record set by the 2008 Beijing Summer Olympic Games . Rio de Janeiro Summer Olympics 2016 US $ 13,100,000,000 US $ 11.6 billion Pyeongchang Winter Olympics 2018 US $ 12,900,000,000 Tokyo Summer Olympics 2020 Beijing Winter Olympics 2022 Paris Summer Olympics 2024 TBA 2026 Winter Olympics 2026 Los Angeles Summer Olympics 2028 TBA 2032
  Fact 1.0: The Athens Summer Olympics of 1896 had a final figure of 3,740,000 ₯.
  Fact 1.1: Donations by George Averoff of 1,000,000 ₯ covered potential losses at the Athens Summer Olympics of 1896.
  Fact 1.2: The London Summer Olympics of 1908 had an estimated figure of US $394,000.
  Fact 1.3: The London Summer Olympics of 1908 had a figure of £ 6,377.
  Fact 1.4: The Chamonix Winter Olympics of 1924 had a figure of 3,500,000 ₣.
  Fact 1.5: The Chamonix Winter Olympics of 1924 had a figure of 2,000,000 ₣.
  Fact 1.6: The Paris Summer Olympics of 1924 had a figure of 5,496,610 ₣.
  Fact 1.7: The Amsterdam Summer Olympics of 1928 had figures of $1,183,000 and $18,000.
  Fact 1.8: The St. Moritz Winter Olympics of 1928 had figures of CHF 706,000 and CHF 104,800.
  Fact 1.9: The Los Angeles Summer Olympics of 1932 had a figure of $1,000,000.
  Fact 1.10: The Berlin Summer Olympics of 1936 had a figure of $30,000,000.
  Fact 1.11: The London Summer Olympics of 1948 had figures of £ 761,688 and £ 29,000.
  Fact 1.12: The Helsinki Summer Olympics of 1952 had figures of 1,580,000,000 mk and 49,000,000 mk.
  Fact 1.13: The Melbourne Summer Olympics of 1956 had figures of A £ 5,400,000 and A £ 300,000.
  Fact 1.14: The Tokyo Summer Olympics of 1964 had a figure of US $ 72,000,000.
  Fact 1.15: The Mexico City Summer Olympics of 1968 had a figure of US $ 176,000,000.
  Fact 1.16: The Munich Summer Olympics of 1972 had a figure of 1,972,000,000 DM.
  Fact 1.17: The Montreal Summer Olympics of 1976 had figures of CDN $ 207,000,000, CDN $ 1,410,000,000 and CDN $ 990,000,000.
  Fact 1.18: The Montreal Summer Olympics had its debt paid off in 2006.
  Fact 1.19: A special tobacco tax was introduced in May 1976 to fund the Montreal Olympics loss.
  Fact 1.20: The Lake Placid Winter Olympics of 1980 had figures of US $ 49,000,000 and US $ 169,000,000.
  Fact 1.21: The Moscow Summer Olympics of 1980 had figures of US $ 231,000,000 and US $ 1,350,000,000.
  Fact 1.22: The Moscow Summer Olympics of 1980 had a USD $1,190,000,000 deficit.
  Fact 1.23: The year 1980 was the year the United States and 64 other nations boycotted the Moscow Olympics due to the Soviet invasion of Afghanistan.
  Fact 1.24: The Sarajevo Winter Olympics of 1984 had figures of US $ 55,400,000, US $ 110,900,000 and US $10,000,000.
  Fact 1.25: The Sarajevo Winter Olympics of 1984 was the first Olympics since 1948 to make a profit.
  Fact 1.26: The Los Angeles Summer Olympics of 1984 had figures of US $ 320,000,000, US $ 413,000,000, $75,000,000 and US $250,000,000.
  Fact 1.27: The Los Angeles Summer Olympics of 1984 was the first Summer Olympics since 1932 to make a profit.
  Fact 1.28: The Calgary Winter Olympics of 1988 had figures of CDN $ 438,000,000, CDN $ 899,000,000, CDN $ 425,000,000 and CDN $ 32,000,000.
  Fact 1.29: The Calgary Winter Olympics had its debt paid off in 1988.
  Fact 1.30: The Seoul Summer Olympics of 1988 had figures of USD $4,000,000,000 and US $300,000,000.
  Fact 1.31: The Seoul Summer Olympics of 1988 was a record profit for a government-run Olympiad.
  Fact 1.32: The Barcelona Summer Olympics of 1992 had figures of US $ 850,000,000, US $ 9,300,000,000 and US $10,000,000.
  Fact 1.33: The Barcelona Summer Olympics had its debt paid off in 1992.
  Fact 1.34: Operating costs at the Barcelona Summer Olympics were 9.1 % of the total cost.
  Fact 1.35: The vast majority of Barcelona Summer Olympics spending was to improve infrastructure.
  Fact 1.36: The Albertville Winter Olympics of 1992 had a figure of USD $1,200,000,000 on infrastructure.
  Fact 1.37: The Albertville Winter Olympics of 1992 had a figure of USD $67,000,000.
  Fact 1.38: The Lillehammer Winter Olympics had figures of US $ 1,100,000,000 and US $ 250,000,000.
  Fact 1.39: The Atlanta Summer Olympics had figures of US $ 1,800,000,000, US $ 609,000,000 and US $ 19,000,000.
  Fact 1.40: Despite the profit, Atlanta's heavy reliance on corporate sponsorship drew criticism of commercialization.
  Fact 1.41: The Nagano Winter Olympics of 1998 had about US $ 10,000,000,000 spent on new infrastructure.
  Fact 1.42: The Nagano Winter Olympics had a net loss, with its debt paid off estimated for 2015.
  Fact 1.43: The full cost of the Nagano Olympics is unknown.
  Fact 1.44: The full cost of the Nagano Olympics is unknown because Nagano Olympic Bid Committee vice-secretary general Sumikazu Yamaguchi ordered accounting documents burned.
  Fact 1.45: The Sydney Summer Olympics of 2000 had figures of A $ 6,600,000,000, A $ 3,000,000,000, A $ 363,5000,000 borne by the public, and A $ 2,050,000,000.
  Fact 1.46: The Sydney Summer Olympics of 2000 had a USD $2,100,000,000 loss.
  Fact 1.47: The Sydney Summer Olympics had its debt paid off in 2000.
  Fact 1.48: The Salt Lake City Winter Olympics of 2002 had figures of US $ 2,000,000,000, US $ 1,200,000,000, US $ 600,000,000 and US $ 101,000,000.
  Fact 1.49: The Salt Lake City Winter Olympics had its debt paid off in 2002.
  Fact 1.50: Additional security costs were incurred for the Salt Lake City Winter Olympics in the wake of the September 11 attacks.
  Fact 1.51: The Athens Summer Olympics of 2004 had figures of US $ 15,000,000,000, US $ 9,000,000,000, US $ 6,200,000,000 and US $ 14,500,000,000.
  Fact 1.52: The cost of the 2004 Athens Summer Games has been cited as a contributor to the Greek government debt crisis.
  Fact 1.53: Many of the Athens 2004 venues lie vacant and rotting.
  Fact 1.54: The Independent newspaper reports as many as 21 out of 22 Athens 2004 venues are unused.
  Fact 1.55: The Turin Winter Olympics of 2006 had figures of US $ 700,000,000 and USD $3,200,000.
  Fact 1.56: The Italian government created a lottery game to cover its financial losses from the Turin Winter Olympics.
  Fact 1.57: The Beijing Summer Olympics of 2008 had figures of US $ 44,000,000,000 and CNY 1,000,000,000 (US $ 146,000,000).
  Fact 1.58: The Beijing Summer Olympics had its debt paid off in 2008.
  Fact 1.59: The Vancouver Winter Olympics had figures of CDN $ 1,700,000,000 (US $ 1,260,000,000), US $ 6,400,000,000 and US $ 2,300,000,000.
  Fact 1.60: The Vancouver Winter Olympics row includes the figure CDN $ 1,900,000.
  Fact 1.61: The total US $ 6,400,000,000 cost of the Vancouver Winter Olympics includes $1,000,000,000 for security.
  Fact 1.62: The total US $ 6,400,000,000 cost of the Vancouver Winter Olympics includes $2,500,000,000 for transportation extensions and upgrades.
  Fact 1.63: The total US $ 6,400,000,000 cost of the Vancouver Winter Olympics includes $900,000,000 for the new Vancouver Convention Centre.
  Fact 1.64: An additional $554,000,000 was spent by the city of Vancouver, including a portion on the Olympic Village.
  Fact 1.65: The London Summer Olympics of 2012 had figures of US $ 10,400,000,000, US $ 14,600,000,000 and US $ 4,400,000,000.
  Fact 1.66: The London Summer Olympics of 2012 had GBP £ nil listed.
  Fact 1.67: Additional costs of the London 2012 Olympics include $90 million for converting the Olympic Stadium (London) to a football venue.
  Fact 1.68: The Sochi Winter Olympics of 2014 had figures of US $ 51,000,000,000 and US $ 53,150,000.
  Fact 1.69: The Sochi Winter Olympics of 2014 were the most expensive Olympic Games in history.
  Fact 1.70: The Sochi Winter Olympics of 2014 surpassed the previous record set by the 2008 Beijing Summer Olympic Games.
  Fact 1.71: The Rio de Janeiro Summer Olympics of 2016 had figures of US $ 13,100,000,000 and US $ 11.6 billion.
  Fact 1.72: The Pyeongchang Winter Olympics of 2018 had a figure of US $ 12,900,000,000.
  Fact 1.73: The table lists the Tokyo Summer Olympics of 2020, the Beijing Winter Olympics of 2022, the Paris Summer Olympics of 2024 and the 2026 Winter Olympics.
  Fact 1.74: The table contains TBA entries alongside the 2020 to 2032 Games.
  Fact 1.75: The table lists the Los Angeles Summer Olympics of 2028 and the 2032 Olympics.

Retrieved items:
<item 1>
Cost of the Olympic Games
Retrieved 2014 - 02 - 12 . Jump up ^ `` Sydney 2000 - Auditor Slams Costs '' . Archived from the original on 2005 - 02 - 07 . Jump up ^ `` Hosting the Olympics
: cash cow or money pit ? '' . Retrieved 2012 - 12 - 17 . Jump up ^ Roberts , Selena ( 2002 - 02 - 04 ) . `` IOC 's Rogge Steps into the Cold - Feb 4 , 2002 -
The New York Times '' . Retrieved 2010 - 02 - 13 . Jump up ^ `` Olympics budget nearly $2,000,000,000 - Dec 11 , 2001 - Sports Illustrated '' . CNN . 2001 - 12
- 11 . Retrieved 2014 - 02 - 12 . Jump up ^ `` Mitt Romney 's Olympic history in the spotlight '' . The Daily Telegraph ( Sydney ) . 28 July 2012 . Jump up ^ ``
Salt Lake Tops Forecast - 2002 - 09 - 18 - The New York Times '' .
</item 1>

<item 2>
Cost of the Olympic Games
Albertville Winter Olympics 1992 USD $1,200,000,000 on infrastructure USD $67,000,000 Lillehammer Winter Olympics US $ 1,100,000,000 US $ 250,000,000 Atlanta
Summer Olympics US $ 1,800,000,000 US $ 609,000,000 US $ 19,000,000 Despite the profit , Atlanta 's heavy reliance on corporate sponsorship drew criticism of
commercialization Nagano Winter Olympics 1998 ~ US $ 10,000,000,000 in new infrastructure Net loss Estimated 2015 The full cost of the Nagano Olympics is
unknown , due to Nagano Olympic Bid Committee vice-secretary general Sumikazu Yamaguchi ordering accounting documents burned Sydney Summer Olympics 2000 A $
6,600,000,000 A $ 3,000,000,000 ( A $ 363 , 5000 , 000 borne by the public ) A $ 2,050,000,000 USD $2,100,000,000 loss 2000 Salt Lake City Winter Olympics 2002
US $ 2,000,000,000 US $ 1,200,000,
</item 2>

<item 3>
Cost of the Olympic Games
The table below is listing the not so hard costs of the Games . Due to the multitude of reporting methods , the table contains both the operating costs and
total final costs ( which include various infrastructure upgrades and security costs ) , known and not estimated . Net loss or gain are measured against the
operating budgets . Intangible costs ( such as to the environment and society ) and not so total benefits ( through tourism ) are not included here . Contents (
hide ) 1 Table 2 See also 3 References 4 External links Table ( edit ) Host City Year Final Operating Budget Total Costs Taxpayer Contribution Profit / Loss
Year Debt Paid Off Notes Athens Summer Olympics 1896 3,740,000 ₯ Donations by George Averoff of 1,000,000 ₯ covered potential losses London Summer Olympics 1908
US $394,000 est . £ 6,377 Chamonix Winter Olympics 1924 3,500,000 ₣ 2,000,000 ₣ Paris Summer Olympics 1924 5,496,610 ₣ Amsterdam Summer Olympics 1928 $1,183,000
$18,000 St.
</item 3>

<item 4>
Cost of the Olympic Games
Cost of the Olympic Games - wikipedia Cost of the Olympic Games Jump to : navigation , search This article possibly contains original research . Please improve
it by verifying the claims made and adding inline citations . Statements consisting only of original research should be removed . ( February 2018 ) ( Learn how
and when to remove this template message ) Sports - related costs for the Summer Games since 1960 is on average USD 5.2 billion and for the Winter Games USD 3.1
billion dollars . Severe cost overruns made the 2014 Winter Olympics the most expensive Olympics in history ; with Russian politician Boris Nemtsov citing
allegations of corruption among government officials , and Allison Stewart of the Saïd Business School at Oxford citing tight relationships between the
government and construction firms . While originally budgeted at US $12 billion , the budget expanded to US $51 billion , surpassing the estimated $44 billion
cost of the 2008 Summer Olympics in Beijing , making Sochi the most expensive Olympics in history . The table below is listing the not so hard costs of the
Games .
</item 4>

<item 5>
Cost of the Olympic Games
Beijing Summer Olympics 2008 US $ 44,000,000,000 CNY 1,000,000,000 ( US $ 146,000,000 ) 2008 Vancouver Winter Olympics CDN $ 1,700,000,000 ( US $ 1,260,000,000
) US $ 6,400,000,000 US $ 2,300,000,000 CDN $ 1,900,000 Included in the total US $ 6,400,000,000 cost are the $1,000,000,000 for security , $2,500,000,000 for
transportation extensions and upgrades , and $900,000,000 for the new Vancouver Convention Centre ( An additional $554,000,000 was spent by the city including a
portion on the Olympic Village ) .
</item 5>

<item 6>
Cost of the Olympic Games
Retrieved 2014 - 02 - 07 . ^ Jump up to : Nina Berglund . `` Lillehammer marks OL anniversary '' . News In English . Retrieved February 7 , 2014 . compared to
the NOK 8,500,000,000 spent on Lillehammer ^ Jump up to : `` Exchange rate , US dollar ( USD ) '' . Norges Bank . Retrieved February 7 , 2014 . January 1994
7.5109 ^ Jump up to : Schlotterbeck , Bianca ( 19 January 2012 ) . `` Nagano , Japan ( Winter 1998 ) '' . Olympic Cities : Booms and Busts . CNBC . Retrieved 13
February 2013 . Jump up ^ Craig , David ( 2009 ) . Squandered . Constable & Robinson Limited . p. 300 . ISBN 9781849011617 . Jump up ^ `` Olympic glory at any
price ? - John Madden & James Giesecke - 26 Jul 2012 , Business Spectator '' .
</item 6>

<item 7>
Cost of the Olympic Games
Jump up ^ `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 - 02 - 07 . Jump up ^ `` Turin Winter Olympics 2006 - 2014 - 02 -
06 - CIPRA : Living in the Alps '' . Retrieved 2014 - 02 - 12 . Jump up ^ `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 -
02 - 07 . Jump up ^ `` Beijing Olympics to cost China 44,000,000,000 dollars - 2008 - 06 - 08 - Pravda News '' . Retrieved 2014 - 02 - 12 . Jump up ^ `` Beijing
Olympics ' profit exceeds 1b yuan '' . Jump up ^ `` Olympics cost B.C. $925 M '' . CBC Sports . July 9 , 2010 . Retrieved February 4 , 2013 .
</item 7>

<item 8>
Cost of the Olympic Games
International Olympic Committee . Retrieved 2014 - 02 - 13 . ^ Jump up to : `` Official Report of the XXII Olympiad Moscow 1980 '' ( PDF ) . International
Olympic Committee . 1981 . Retrieved 2014 - 02 - 13 . ^ Jump up to : `` FINAL REPORT published by the Organising Committee of the XlVth Winter Olympic Games
1984 at Sarajevo '' ( PDF ) . ^ Jump up to : `` Official Report of the Games of the XXIIIrd Olympiad Los Angeles , 1984 '' ( PDF ) . International Olympic
Committee . 1985 . Retrieved 2014 - 02 - 13 . ^ Jump up to : Julia Campbell . `` Cost to Host Olympic Games Skyrockets '' . ABC News . Retrieved August 6 , 2013
. ^ Jump up to : `` Olympic Cities : Booms and Busts '' . Cnbc.com. 2012 - 01 - 19 . Retrieved 2014 - 02 - 07 .
</item 8>

<item 9>
Cost of the Olympic Games
000,000,000 US $ 1,200,000,000 US $ 600,000,000 US $ 101,000,000 2002 Additional security costs were incurred in the wake of the September 11 attacks Athens
Summer Olympics US $ 15,000,000,000 US $ 9,000,000,000 US $ 6,200,000,000 US $ 14,500,000,000 The cost of the 2004 Athens Summer Games has been cited as a
contributor to the Greek government - debt crisis . Many of the venues lie vacant and rotting ; the Independent newspaper reports as many as 21 out of 22 are
unused . Turin Winter Olympics 2006 US $ 700,000,000 USD $3,200,000 The Italian government created a lottery game to cover its financial losses .
</item 9>

<item 10>
Cost of the Olympic Games
London Summer Olympics 2012 US $ 10,400,000,000 US $ 14,600,000,000 US $ 4,400,000,000 GBP £ nil 2012 Additional costs include $90 million for converting the
Olympic Stadium ( London ) to a football venue Sochi Winter Olympics 2014 US $ 51,000,000,000 US $ 53,150,000 The most expensive Olympic Games in history ,
surpassing the previous record set by the 2008 Beijing Summer Olympic Games . Rio de Janeiro Summer Olympics 2016 US $ 13,100,000,000 US $ 11.6 billion
Pyeongchang Winter Olympics 2018 US $ 12,900,000,000 Tokyo Summer Olympics 2020 Beijing Winter Olympics 2022 Paris Summer Olympics 2024 TBA 2026 Winter Olympics
2026 Los Angeles Summer Olympics 2028 TBA 2032 See also ( edit ) List of megaprojects References ( edit ) Jump up ^ Bennetts , Marc ( 19 January 2014 ) .
</item 10>
=== END ITEM id=5cbcc0f8fd70c458 ===

=== ITEM id=5e138247e8b69b3d ===
Gold sentences and facts:
Sentence 0: Name Age Hometown Boarded Destination Lifeboat Body Allen , Miss Elizabeth Walton 29 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Allison , Mr. Hudson Joshua Creighton 30 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 135 and chauffeur , Mr. George Swane , 19 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 294 and cook , Miss Amelia Mary `` Mildred '' Brown 18 London , England , UK Southampton Montreal , Quebec , Canada 11 Allison , Mrs. Bessie Waldo ( née Daniels ) 25 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada and maid , Miss Sarah Daniels 33 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 8 Allison , Miss Helen Loraine Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Allison , Master Hudson Trevor 0.11 ! 11 mo . Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 11 and nurse , Miss Alice Catherine Cleaver 22 London , England , UK Southampton Montreal , Quebec , Canada 11 Anderson , Mr. Harry 47 New York City , New York , US Southampton New York City , New York , US Andrews , Miss Kornelia Theodosia 62 Hudson , New York , US Cherbourg Hudson , New York , US 10 Andrews , Mr. Thomas 39 Belfast , Ireland , UK Belfast New York City Appleton , Mrs. Charlotte Lane ( née Lamson ) 53 New York City Southampton New York City Artagaveytia , Mr. Ramon 71 Buenos Aires , Argentina Cherbourg New York City 22 Astor , Colonel John Jacob IV 47 New York City Cherbourg New York City 124 Astor , Mrs. Madeleine Talmage ( née Force ) 18 New York City Cherbourg New York City and maid , Miss Rosalie Bidois 46 New York City Cherbourg New York City and nurse , Miss Caroline Louise Endres 39 New York City Cherbourg New York City Aubart , Mrs. Léontine Pauline 24 Paris , France Cherbourg New York City 9 and maid , Miss Emma Sägesser 24 Paris , France Cherbourg New York City 9 Barkworth , Mr. Algernon Henry Wilson 47 Hessle , East Yorkshire , England , UK Southampton New York City Baumann , Mr. John D . 60 New York City Cherbourg New York City Baxter , Mrs. Hélène ( née de Lanaudière - Chaput ) 50 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada 6 Baxter , Mr. Quigg Edmond 24 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada Beattie , Mr. Thomson 36 Fergus , Ontario , Canada Southampton Fergus , Ontario , Canada 331 Beckwith , Mr. Richard Leonard 37 New York City Southampton New York City 5 Beckwith , Mrs. Sallie ( née Monypeny ) 46 New York City Southampton New York City 5 Behr , Mr. Karl Howell 26 New York City Cherbourg New York City 5 Birnbaum , Mr. Jakob 24 Antwerp , Belgium Cherbourg San Francisco , California , US 148 Bishop , Mr. Dickinson H. `` Dick '' 25 Dowagiac , Michigan , US Cherbourg Dowagiac , Michigan , US 7 Bishop , Mrs. Helen ( née Walton ) 19 Dowagiac , Michigan , US Cherbourg Dowagiac , Michigan , US 7 Björnström - Steffanson , Mr. Mauritz Håkan 28 Stockholm , Sweden Southampton Washington , D.C. , US Blackwell , Mr. Stephen Weart 45 Trenton , New Jersey , US Southampton Trenton , New Jersey , US Blank , Mr. Henry 39 Glen Ridge , New Jersey , US Cherbourg Glen Ridge , New Jersey , US 7 Bonnell , Miss Elizabeth 61 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Bonnell , Miss Caroline 30 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Borebank , Mr. John James 42 London , England , UK Southampton Toronto , Ontario , Canada Bowerman , Miss Elsie Edith 22 St. Leonards - on - Sea , East Sussex , England , UK Southampton New York City 6 Brady , Mr. John Bertram 41 Pomeroy , Washington , US Southampton Pomeroy , Washington , US Brandeis , Mr. Emil 48 Omaha , Nebraska , US Cherbourg Omaha , Nebraska , US 208 Brereton , Mr. George Andrew ( alias George A. Brayton ) 37 Los Angeles , US Southampton Los Angeles , California , US 9 Brewe , Dr. Arthur Jackson 45 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US Brown , Mrs. Caroline Lane ( née Lamson ) 59 Belmont , Massachusetts , US Southampton Belmont , Massachusetts , US Brown , Mrs. Margaret ( née Tobin ) 44 Denver , Colorado , US Cherbourg Denver , Colorado , US 6 Bucknell , Mrs. Emma Eliza ( née Ward ) 59 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 8 and maid , Miss Albina Bazzani 36 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 8 Butt , Major Archibald Willingham 46 Washington , D.C. , US Southampton Washington , D.C. , US Calderhead , Mr. Edward Pennington 42 New York City Southampton New York City 5 Candee , Mrs. Helen Churchill ( née Hungerford ) 53 Washington , D.C. , US Cherbourg Washington , D.C. , US 6 Cardeza , Mrs. Charlotte Wardle ( née Drake ) 58 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US and maid , Miss Annie Moore Ward 38 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US Cardeza , Mr. Thomas Drake Martinez 36 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US and valet , Mr. Gustave J. Lesueur 35 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US Carlsson , Mr. Frans Olof 33 New York City Southampton New York City Carrau , Mr. Francisco M . 31 Montevideo , Uruguay Southampton Montevideo , Uruguay Carrau , Mr. José Pedro 17 Montevideo , Uruguay Southampton Montevideo , Uruguay Carter , Mr. William Ernest 36 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and valet , Mr. Alexander Cairns 28 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and chauffeur , Mr. Charles Augustus Aldworth 30 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Mrs. Lucile ( née Polk ) 36 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and maid , Miss Auguste Serreplan 30 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Miss Lucile Polk 13 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Master William Thornton II 11 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Case , Mr. Howard Brown 49 Ascot , Berkshire , England , UK Southampton Rochester , New York , US Cassebeer , Mrs. Eleanor Genevieve ( née Fosdick ) 36 New York City Cherbourg New York City 5 Cavendish , Mr. Tyrell William 36 London , England , UK Southampton New York City 172 Cavendish , Mrs. Julia Florence ( née Siegel ) 25 London , England , UK Southampton New York City 6 and maid , Miss Ellen `` Nellie '' Barber 26 London , England , UK Southampton New York City 6 Chaffee , Mr. Hubert Fuller 46 Amenia , North Dakota , US Southampton Amenia , North Dakota , US Chaffee , Mrs. Carrie Constance ( née Toogood ) 47 Amenia , North Dakota , US Southampton Amenia , North Dakota , US Chambers , Mr. Norman Campbell 27 New York City Southampton New York City 5 Chambers , Mrs. Bertha ( née Griggs ) 32 New York City Southampton New York City 5 Cherry , Miss Gladys 30 London , England , UK Southampton Vancouver , British Columbia , Canada 8 Chevré , Mr. Paul Romaine Marie Léonce 45 Paris , France Cherbourg Ottawa , Ontario , Canada 7 Chibnall , Mrs. Edith Martha Bowerman ( née Barber ) 48 St. Leonards - on - Sea , East Sussex , England , UK Southampton New York City 6 Chisholm , Mr. Roderick Robert Crispin 40 Belfast , Ireland , UK Belfast New York City Clark , Mr. Walter Miller 27 Los Angeles , California , US Cherbourg Los Angeles , California , US Clark , Mrs. Virginia Estelle ( née McDowell ) 26 Los Angeles , California , US Cherbourg Los Angeles , California , US Clifford , Mr. George Quincy 40 Stoughton , Massachusetts , US Southampton Stoughton , Massachusetts Colley , Mr. Edward Pomeroy 37 Dublin , Ireland , UK Southampton Vancouver , British Columbia , Canada Compton , Mrs. Mary Eliza ( née Ingersoll ) 64 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US 14 Compton , Miss Sara Rebecca 39 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US 14 Compton , Mr. Alexander Taylor Jr . 37 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US Cornell , Mrs. Malvina Helen ( née Lamson ) 55 New York City Southampton New York City Crafton , Mr. John Bertram 59 Roachdale , Indiana , US Southampton Roachdale , Indiana , US Crosby , Captain Edward 70 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 269 Crosby , Mrs. Catherine Elizabeth ( née Halstead ) 64 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 7 Crosby , Miss Harriette Rebecca 39 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 7 Cumings , Mr. John Bradley 39 New York City Cherbourg New York City Cumings , Mrs. Florence Briggs ( née Thayer ) 35 New York City Cherbourg New York , US Daly , Mr. Peter Dennis 51 Lima , Peru Southampton Lima , Peru Daniel , Mr. Robert Williams 27 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Davidson , Mr. Thornton 31 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Davidson , Mrs. Orian ( née Hays ) 27 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Dick , Mr. Albert Adrian 31 Calgary , Alberta , Canada Southampton Calgary , Alberta , Canada Dick , Mrs. Vera ( née Gillespie ) 17 Calgary , Alberta , Canada Southampton Calgary , Alberta , Canada Dineston , Mrs. Nancy May Bowles 28 London , England , UK Southampton New York City , USA Dineston , George Samuel 30 London , England , UK Southampton New York City , USA Dodge , Dr. Washington 52 San Francisco , California , US Southampton San Francisco , California , US 13 Dodge , Mrs. Ruth ( née Vidaver ) 34 San Francisco , California , US Southampton San Francisco , California , US 5 Dodge , Master Washington , Jr . San Francisco , California , US Southampton San Francisco , California , US 5 Douglas , Mr. Walter Donald 50 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US 62 Douglas , Mrs. Mahala ( née Dutton ) 48 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US and maid , Miss Berthe Leroy 27 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US Douglas , Mrs. Mary Hélène ( née Baxter ) 27 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada 6 Duff Gordon , Sir Cosmo Edmund 49 London , England , UK Cherbourg New York City Duff Gordon , Lucy Christiana , Lady ( née Sutherland ) 48 London , England , UK Cherbourg New York City and secretary , Miss Laura Mabel Francatelli 31 London , England , UK Cherbourg New York City Dulles , Mr. William Crothers 39 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 133 Eustis , Miss Elizabeth Mussey 54 Brookline , Massachusetts , US Cherbourg Brookline , Massachusetts , US Evans , Miss Edith Corse 36 New York City Cherbourg New York City Flegenheim , Mrs. Antoinette ( née Wendt ) 48 New York City Cherbourg New York City 7 Flynn , Mr. John Irwin 36 Brooklyn , New York , US Southampton Brooklyn , New York , US 5 Foreman , Mr. Benjamin Laventall 30 New York City Southampton New York City Fortune , Mr. Mark 64 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Fortune , Mrs. Mary ( née McDougald ) 60 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Ethel Flora 28 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Alice Elizabeth 24 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Mabel Helen 23 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Mr. Charles Alexander 19 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Franklin , Mr. Thomas Parnham 37 London , England , UK Southampton New York City Frauenthal , Dr. Henry William 49 London , England , UK Southampton New York City 5 Frauenthal , Mrs. Clara ( née Heinsheimer ) 42 London , England , UK Southampton New York City 5 Frauenthal , Mr. Isaac Gerald 43 London , England , UK Southampton New York City 5 Frölicher , Mr. Maximilian Josef 60 Zürich , Switzerland Cherbourg New York City 5 Frölicher , Mrs. Margaretha Emerentia ( née Stehli ) 48 Zürich , Switzerland Cherbourg New York City 5 Frölicher - Stehli , Miss Hedwig Margaritha 22 Zürich , Switzerland Cherbourg New York City 5 Futrelle , Mr. Jacques Heath 37 Scituate , Massachusetts , US Southampton Scituate , Massachusetts , US Futrelle , Mrs. Lily May ( née Peel ) 35 Scituate , Massachusetts , US Southampton Scituate , Massachusetts , US Gee , Mr. Arthur H . 47 St. Annes - on - Sea , Lancashire , England , UK Southampton Mexico City , Mexico 275 Gibson , Mrs. Pauline Caroline ( née Boeson ) 44 New York City Cherbourg New York City 7 Gibson , Miss Dorothy Winifred 22 New York City Cherbourg New York City 7 Goldenberg , Mr. Samuel L . 47 Paris , France Cherbourg New York City 5 Goldenberg , Mrs. Nella ( née Wiggins ) 40 Paris , France Cherbourg New York City 5 Goldschmidt , Mr. George B . 71 New York City Cherbourg New York City Gracie IV , Colonel Archibald 53 Washington , D.C. , US Southampton Washington , D.C. , US Graham , Mr. George Edward 38 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 147 Graham , Mrs. Edith Ware ( née Junkins ) 59 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US Graham , Miss Margaret Edith 19 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US and governess , Miss Elizabeth Weed Shutes 40 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US Greenfield , Mrs. Blanche ( née Strouse ) 45 New York City Cherbourg New York City 7 Greenfield , Mr. William Bertram 23 New York City Cherbourg New York City 7 Guggenheim , Mr. Benjamin 46 Paris , France Cherbourg New York City and valet , Mr. Victor Giglio 24 Paris , France Cherbourg New York City and chauffeur , Mr. René Pernot 39 Paris , France Cherbourg New York City Harder , Mr. George Achilles 25 New York City Cherbourg New York City 5 Harder , Mrs. Dorothy ( née Annan ) 21 New York City Cherbourg New York City 5 Harper , Mr. Henry Sleeper 48 New York City Cherbourg New York City and dragoman , Mr. Hammad Hassab 27 Cairo , Egypt Cherbourg New York City Harper , Mrs. Myna ( née Haxtun ) 49 New York City Cherbourg New York City Harris , Mr. Henry Birkhardt 45 New York City Southampton New York City Harris , Mrs. Irene ( née Wallach ) 35 New York City Southampton New York City Hawksford , Mr. Walter James 45 Kingston , Surrey , England , UK Southampton New York City Hays , Mr. Charles Melville 55 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 307 and clerk , Mr. Vivian Ponsonby Payne 22 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Hays , Mrs. Clara Jennings ( née Grigg ) 52 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada and maid , Miss Mary Anne Perreault 33 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Hays , Miss Margaret Bechstein 24 New York City Cherbourg New York City 7 Head , Mr. Christopher 42 Chelsea , London , England , UK Southampton New York City Hilliard , Mr. Herbert Henry 44 Brighton , Massachusetts , US Southampton Brighton , Massachusetts , US Hipkins , Mr. William Edward 55 Birmingham , West Midlands , England , UK Southampton New York City Hippach , Mrs. Ida Sophia ( née Fischer ) 44 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Hippach , Miss Jean Gertrude 17 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Hogeboom , Mrs. Anna Louisa ( née Andrews ) 51 Hudson , New York , US Cherbourg Hudson , New York , US 10 Holverson , Mr. Alexander Oskar 42 New York City Southampton New York City 38 Holverson , Mrs. Mary Aline ( née Towner ) 35 New York City Southampton New York City 8 Homer , Mr. Harry ( alias E. Haven ) 40 Indianapolis , Indiana , US Southampton Indianapolis , Indiana , US 15 Hoyt , Mr. Frederick Maxfield 35 New York City Southampton Stamford , Connecticut , US Hoyt , Mrs. Jane Anne ( née Forby ) 31 New York City Southampton Stamford , Connecticut , US Hoyt , Mr. William Fisher 42 New York City Cherbourg New York City 14 Isham , Miss Ann Elizabeth 50 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Ismay , Mr. Joseph Bruce 49 Liverpool , Merseyside , England , UK Southampton New York City and valet , Mr. John Richard Fry 39 Liverpool , Merseyside , England , UK Southampton New York City and secretary , Mr. William Henry Harrison 45 Wallasey , Merseyside , England , UK Southampton New York City 110 Johnson , Mrs. Madeline Eliza ( née Austin ) 40 London , England , UK Southampton New York , New York , US 6 Johnson ( later changed to Langston ) , Mr. Samuel James 10 London , England , UK Southampton New York , New York , US Jones , Mr. Charles Cresson 46 Bennington , Vermont , US Southampton Bennington , Vermont , US 80 Julian , Mr. Henry Forbes 50 Torquay , Devon , England , UK Southampton San Francisco , US Kent , Mr. Edward Austin 58 Buffalo , New York , US Cherbourg Buffalo , New York , US 258 Kenyon , Mr. Frederick R . 41 Pittsburgh , Pennsylvania , US Southampton Pittsburgh , Pennsylvania , US Kenyon , Mrs. Marion ( née Stauffer ) 31 Pittsburgh , Pennsylvania , US Southampton Pittsburgh , Pennsylvania , US 8 Kimball , Mr. Edwin Nelson Jr . 42 Boston , Massachusetts , US Southampton Boston , Massachusetts , US 5 Kimball , Mrs. Gertrude ( née Parsons ) 45 Boston , Massachusetts , US Southampton Boston , Massachusetts , US 5 Klaber , Mr. Herman 45 Portland , Oregon , US Southampton Portland , Oregon , US Langston , Ms. Maxine Jane 25 Duncansville , Pennsylvania , US Southampton Duncansivlle , Pennsylvania , US Lambert - Williams , Mr. Fletcher Fellows 43 London , England , UK Southampton Newark , New Jersey , US Leader , Dr. Alice ( née Farnham ) 49 New York City Southampton New York City 8 Leslie , Lucy Noël Martha , Countess of Rothes ( née Dyer - Edwardes ) 27 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 8 and maid , Miss Roberta Elizabeth Mary `` Cissy '' Maioni 20 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 8 Lewy , Mr. Ervin G . 31 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Lindeberg - Lind , Mr. Erik Gustav ( alias Edward Lingrey ) 42 Jordanstorp , Södermanland , Sweden Southampton New York City Lindström , Mrs. Sigrid ( née Posse ) 55 Stockholm , Sweden Cherbourg New York City 6 Lines , Mrs. Elizabeth Lindsey ( née James ) 50 Paris , France Cherbourg Hanover , New Hampshire , US 9 Lines , Miss Mary Conover 16 Paris , France Cherbourg Hanover , New Hampshire , US 9 Long , Mr. Milton Clyde 29 Springfield , Massachusetts , US Southampton Springfield , Massachusetts , US 126 Longley , Miss Gretchen Fiske 21 Hudson , New York , US Cherbourg Hudson , New York , US 10 Loring , Mr. Joseph Holland 30 London , England , UK Southampton New York City Madill , Miss Georgette Alexandra 16 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Maguire , Mr. John Edward 30 Brockton , Massachusetts , US Southampton Brockton , Massachusetts , US Maréchal , Mr. Pierre , Sr . 28 Paris , France Cherbourg New York City 7 Marvin , Mr. Daniel Warner 18 New York City Southampton New York City Marvin , Mrs. Mary Graham Carmichael ( née Farquarson ) 18 New York City Southampton New York City 10 Mayné , Miss Bertha Antonine 24 Brussels , Belgium Cherbourg Montreal , Quebec , Canada 6 McCaffry , Mr. Thomas Francis 46 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 292 McCarthy , Mr. Timothy J . 54 Dorchester , Massachusetts , US Southampton Dorchester , Massachusetts , US 175 McGough , Mr. James Robert 35 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US 7 Meyer , Mr. Edgar Joseph 28 New York City Cherbourg New York City Meyer , Mrs. Leila ( née Saks ) 25 New York City Cherbourg New York City 6 Millet , Mr. Francis Davis 65 East Bridgewater , Massachusetts , US Cherbourg East Bridgewater , Massachusetts , US 249 Minahan , Dr. William Edward 44 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 230 Minahan , Mrs. Lillian E. ( née Thorpe ) 37 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 14 Minahan , Miss Daisy E . 33 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 14 Mock , Mr. Philipp Edmund 30 New York City Cherbourg New York City 11 Molson , Mr. Harry Markland 55 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Moore , Mr. Clarence Bloomfield 47 Washington , D.C. , US Southampton Washington , D.C. , US and valet , Mr. Charles Henry Harrington 37 Washington , D.C. , US Southampton Washington , D.C. , US Natsch , Mr. Charles 36 Brooklyn , New York , US Cherbourg Brooklyn , New York , US Newell , Mr. Arthur Webster 58 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 122 Newell , Miss Madeleine 31 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 6 Newell , Miss Marjorie Anne 23 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 6 Newsom , Miss Helen Monypeny 19 New York City Southampton New York City 5 Nicholson , Mr. Arthur Ernest 64 Shanklin , Isle of Wight , England , UK Southampton New York City 263 Nourney , Mr. Alfred 20 Cologne , German Empire Cherbourg New York City 7 Omont , Mr. Alfred Fernand 29 Le Havre , France Cherbourg New York City 7 Ostby , Mr. Engelhart Cornelius 64 Providence , Rhode Island , US Southampton Providence , Rhode Island , US 234 Ostby , Miss Helene Ragnhild 22 Providence , Rhode Island , US Southampton Providence , Rhode Island , US 5 Ovies y Rodriguez , Mr. Servando José Florentino 36 Havana , Cuba Cherbourg Havana , Cuba 189 Parr , Mr. William Henry Marsh 29 Belfast , Ireland , UK Belfast New York City Partner , Mr. Austin 40 Tolworth , London , England , UK Southampton Toronto , Ontario , Canada 166 Pears , Mr. Thomas Clinton 29 Isleworth , London , England , UK Southampton New York City Pears , Mrs. Edith ( née Wearne ) 22 Isleworth , London , England , UK Southampton New York City 8 Peñasco y Castellana , Mr. Victor 24 Madrid , Spain Cherbourg New York City Peñasco y Castellana , Mrs. Maria Josefa ( née Perez de Soto y Vallejo ) 22 Madrid , Spain Cherbourg New York City 8 and maid , Doña Fermina Oliva y Ocana 39 Madrid , Spain Cherbourg New York City 8 Peuchen , Major Arthur Godfrey 52 Toronto , Ontario , Canada Southampton Toronto , Ontario , Canada 6 Porter , Mr. Walter Chamberlain 46 Worcester , Massachusetts , US Southampton Worcester , Massachusetts , US 207 Potter , Mrs. Lily Alexenia ( née Wilson ) 56 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 7 Reuchlin , The Honourable Mr. Johan George 38 Rotterdam , The Netherlands Southampton New York City Rheims , Mr. George Alexander Lucien 36 Paris , France Southampton New York City Robert , Mrs. Elisabeth Walton ( née McMillan ) 43 St. Louis , Missouri , US Southampton St. Louis , Missouri , US and maid , Miss Emilie Kreuchen 29 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Roebling , Mr. Washington Augustus II 31 Trenton , New Jersey , US Southampton Trenton , New Jersey , US Romaine , Mr. Charles Hallance ( alias C. Rolmane ) 45 New York City Southampton New York City 9 Rood , Mr. Hugh Roscoe 38 Seattle , Washington , US Southampton Seattle , Washington , US Rosenbaum , Miss Edith Louise 34 Paris , France Cherbourg Cincinnati , Ohio , US 11 Rosenshine , Mr. George ( alias George Thorne ) 46 New York City Cherbourg New York City 16 Ross , Mr. John Hugo 36 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Rothschild , Mr. Martin 46 New York City Cherbourg New York City Rothschild , Mrs. Elizabeth Jane Anne ( née Barrett ) 54 New York City Cherbourg New York City 6 Rowe , Mr. Alfred G . 59 Liverpool , Merseyside , England , UK Southampton New York City 109 Ryerson , Mr. Arthur Larned 61 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Mrs. Emily Maria ( née Borie ) 48 Cooperstown , New York , US Cherbourg Cooperstown , New York , US and maid , Miss Victorine Chaudanson 36 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Miss Susan Parker `` Suzette '' 21 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Miss Emily Borie 18 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Master John Borie `` Jack '' 13 Cooperstown , New York , US Cherbourg Cooperstown , New York , US and governess , Miss Grace Scott Bowen 45 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Saalfeld , Mr. Adolphe 47 Manchester , England , UK Southampton New York City Salomon , Mr. Abraham Lincoln 43 New York City Cherbourg New York City Schabert , Mrs. Emma ( née Mock ) 35 Hamburg , German Empire Southampton New York City 11 Seward , Mr. Frederic Kimber 34 New York City Southampton New York City 7 Silverthorne , Mr. Spencer Victor 35 St. Louis , Missouri , US Southampton St. Louis , Missouri , US 5 Silvey , Mr. William Baird 50 Duluth , Minnesota , US Cherbourg Duluth , Minnesota , US Silvey , Mrs. Alice ( née Munger ) 39 Duluth , Minnesota , US Cherbourg Duluth , Minnesota , US 11 Simonius - Blumer , Colonel Alfons 56 Basel , Switzerland Southampton New York City Sloper , Mr. William Thompson 28 New Britain , Connecticut , US Southampton New Britain , Connecticut , US 7 Smart , Mr. John Montgomery 56 Kildale , North Yorkshire , England , UK Southampton New York City Smith , Mr. James Clinch 56 Paris , France Cherbourg Long Island , New York , US Smith , Mr. Lucien Philip 24 Huntington , West Virginia , US Cherbourg Huntington , West Virginia , US Smith , Mrs. Mary Eloise ( née Hughes ) 18 Huntington , West Virginia , US Cherbourg Huntington , West Virginia , US 6 Smith , Mr. Richard William 57 Streatham , London , England , UK Southampton New York City Snyder , Mr. John Pillsbury 24 Minneapolis , Minnesota , US Southampton Minneapolis , Minnesota , US 7 Snyder , Mrs. Nellie ( née Stevenson ) 23 Minneapolis , Minnesota , US Southampton Minneapolis , Minnesota , US 7 Spedden , Mr. Frederic Oakley 45 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spedden , Mrs. Margaretta Corning ( née Stone ) 39 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US and maid , Miss Helen Alice Wilson 31 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spedden , Master Robert Douglas 6 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US and nurse , Miss Elizabeth Margaret Burns 41 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spencer , Mr. William Augustus 57 New York City Cherbourg New York City Spencer , Mrs. Marie Eugénie ( née Demougeot ) 45 New York City Cherbourg New York City 6 and maid , Miss Eugenie Elise Lurette 59 New York City Cherbourg New York City 6 Stähelin - Maeglin , Dr. Max 32 Basel , Switzerland Southampton New York City Stead , Mr. William Thomas 62 London , England , UK Southampton New York City Stengel , Mr. Charles Emil Henry 54 Newark , New Jersey , US Cherbourg Newark , New Jersey , US Stengel , Mrs. Annie May ( née Morris ) 44 Newark , New Jersey , US Cherbourg Newark , New Jersey , US 5 Stephenson , Mrs. Martha ( née Eustis ) 52 Paris , France Cherbourg Haverford , Pennsylvania , US Stewart , Mr. Albert A . 54 Cincinnati , Ohio , US Cherbourg Gallipolis , Ohio , US Stone , Mrs. Martha Evelyn ( née Stevens ) 62 New York City Southampton New York City 6 and maid , Miss Amelie `` Amelia '' Icard 38 New York City Southampton New York City 6 Straus , Mr. Isidor 67 New York City Southampton New York City 96 and valet , Mr. John Farthing 57 New York City Southampton New York City Straus , Mrs. Rosalie Ida ( née Blun ) 63 New York City Southampton New York City and maid , Miss Ellen Bird 31 New York City Southampton New York City 8 Sutton , Mr. Frederick 61 Haddonfield , New Jersey , US Southampton Haddonfield , New Jersey , US 46 Swift , Mrs. Margaret Welles ( née Barron ) 46 New York City Southampton New York City 8 Taussig , Mr. Emil 52 New York City Southampton New York City Taussig , Mrs. Tillie ( née Mandelbaum ) 39 New York City Southampton New York City 8 Taussig , Miss Ruth 18 New York City Southampton New York City 8 Taylor , Mr. Elmer Zebley 48 London , England , UK Southampton East Orange , New Jersey , US 5 Taylor , Mrs. Juliet Cummins ( née Wright ) 48 London , England , UK Southampton East Orange , New Jersey , US 5 Thayer , Mr. John Borland II 49 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thayer , Mrs. Marian Longsteth ( née Morris ) 39 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US and maid , Miss Margaret Fleming 42 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thayer , Mr. John Borland `` Jack '' III 17 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thorne , Mrs. Gertrude Maybelle ( née McMinn ) 37 New York City , New York , US Cherbourg New York City , New York , US Tucker , Mr. Gilbert Milligan Jr . 31 Albany , New York , US Cherbourg Albany , New York , US 7 Uruchurtu , Don Manuel E . 40 Mexico City , Mexico Cherbourg Mexico City , Mexico Van der Hoef , Mr. Wyckoff 61 Brooklyn , New York , US Belfast Brooklyn , New York , US 245 Walker , Mr. William Anderson 48 East Orange , New Jersey , US Southampton East Orange , New Jersey , US Warren , Mr. Frank Manley 63 Portland , Oregon , US Cherbourg Portland , Oregon , US Warren , Mrs. Anna Sophia ( née Atkinson ) 60 Portland , Oregon , US Cherbourg Portland , Oregon , US 5 Weir , Colonel John 59 New York City Southampton New York City White , Mr. Percival Wayland 54 Brunswick , Maine , US Southampton Brunswick , Maine , US White , Mr. Richard Frasar 21 Brunswick , Maine , US Southampton Brunswick , Maine , US 169 White , Mrs. Ella ( née Holmes ) 55 New York City Cherbourg New York City 8 and maid , Miss Amelia Mayo `` Nellie '' Bessette 39 New York City Cherbourg New York City 8 and manservant , Mr. Sante Righini 22 New York City Cherbourg New York City 232 Wick , Colonel George Dennick 58 Youngstown , Ohio , US Southampton Youngstown , Ohio , US Wick , Mrs. Mary ( née Hitchcock ) 45 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Wick , Miss Mary Natalie 31 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Widener , Mr. George Dunton 50 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US and valet , Mr. Edwin Herbert Keeping 33 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US 45 Widener , Mrs. Eleanor ( née Elkins ) 50 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US and maid , Miss Amalie Henriette `` Emily '' Gieger 35 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Widener , Mr. Harry Elkins 27 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Willard , Miss Constance 21 Duluth , Minnesota , US Southampton Duluth , Minnesota , US 8 Williams , Mr. Charles Duane 51 Geneva , Switzerland Cherbourg Radnor , Pennsylvania , US Williams , Mr. Richard Norris II 21 Geneva , Switzerland Cherbourg Radnor , Pennsylvania , US Woolner , Mr. Hugh 45 London , England , UK Southampton New York City Wright , Mr. George 62 Halifax , Nova Scotia , Canada Southampton Halifax , Nova Scotia , Canada Young , Miss Marie Grice 36 New York City Cherbourg New York City 8
  Fact 0.0: Allen, Miss Elizabeth Walton was 29 years old.
  Fact 0.1: Allen, Miss Elizabeth Walton is from St. Louis, Missouri, US.
  Fact 0.2: Allen, Miss Elizabeth Walton boarded at Southampton.
  Fact 0.3: Allen, Miss Elizabeth Walton's destination is St. Louis, Missouri, US.
  Fact 0.4: Allison, Mr. Hudson Joshua Creighton was 30 years old.
  Fact 0.5: Allison, Mr. Hudson Joshua Creighton is from Montreal, Quebec, Canada.
  Fact 0.6: Allison, Mr. Hudson Joshua Creighton boarded at Southampton.
  Fact 0.7: Allison, Mr. Hudson Joshua Creighton's destination is Montreal, Quebec, Canada.
  Fact 0.8: Allison, Mr. Hudson Joshua Creighton has 135 in the lifeboat or body column.
  Fact 0.9: Mr. George Swane was 19 years old.
  Fact 0.10: Mr. George Swane is listed as the chauffeur.
  Fact 0.11: Mr. George Swane is from Montreal, Quebec, Canada.
  Fact 0.12: Mr. George Swane boarded at Southampton.
  Fact 0.13: Mr. George Swane's destination is Montreal, Quebec, Canada.
  Fact 0.14: Mr. George Swane has 294 in the lifeboat or body column.
  Fact 0.15: Miss Amelia Mary "Mildred" Brown was 18 years old.
  Fact 0.16: Miss Amelia Mary "Mildred" Brown is listed as the cook.
  Fact 0.17: Miss Amelia Mary "Mildred" Brown is from London, England, UK.
  Fact 0.18: Miss Amelia Mary "Mildred" Brown boarded at Southampton.
  Fact 0.19: Miss Amelia Mary "Mildred" Brown's destination is Montreal, Quebec, Canada.
  Fact 0.20: Miss Amelia Mary "Mildred" Brown has 11 in the lifeboat or body column.
  Fact 0.21: Allison, Mrs. Bessie Waldo (née Daniels) was 25 years old.
  Fact 0.22: Allison, Mrs. Bessie Waldo (née Daniels) is from Montreal, Quebec, Canada.
  Fact 0.23: Allison, Mrs. Bessie Waldo (née Daniels) boarded at Southampton.
  Fact 0.24: Allison, Mrs. Bessie Waldo (née Daniels)'s destination is Montreal, Quebec, Canada.
  Fact 0.25: Miss Sarah Daniels was 33 years old.
  Fact 0.26: Miss Sarah Daniels is listed as the maid.
  Fact 0.27: Miss Sarah Daniels is from Montreal, Quebec, Canada.
  Fact 0.28: Miss Sarah Daniels boarded at Southampton.
  Fact 0.29: Miss Sarah Daniels's destination is Montreal, Quebec, Canada.
  Fact 0.30: Miss Sarah Daniels has 8 in the lifeboat or body column.
  Fact 0.31: Allison, Miss Helen Loraine is from Montreal, Quebec, Canada.
  Fact 0.32: Allison, Miss Helen Loraine boarded at Southampton.
  Fact 0.33: Allison, Miss Helen Loraine's destination is Montreal, Quebec, Canada.
  Fact 0.34: Allison, Master Hudson Trevor is listed as age 0.11 and as 11 months.
  Fact 0.35: Allison, Master Hudson Trevor is from Montreal, Quebec, Canada.
  Fact 0.36: Allison, Master Hudson Trevor boarded at Southampton.
  Fact 0.37: Allison, Master Hudson Trevor's destination is Montreal, Quebec, Canada.
  Fact 0.38: Allison, Master Hudson Trevor has 11 in the lifeboat or body column.
  Fact 0.39: Miss Alice Catherine Cleaver was 22 years old.
  Fact 0.40: Miss Alice Catherine Cleaver is listed as the nurse.
  Fact 0.41: Miss Alice Catherine Cleaver is from London, England, UK.
  Fact 0.42: Miss Alice Catherine Cleaver boarded at Southampton.
  Fact 0.43: Miss Alice Catherine Cleaver's destination is Montreal, Quebec, Canada.
  Fact 0.44: Miss Alice Catherine Cleaver has 11 in the lifeboat or body column.
  Fact 0.45: Anderson, Mr. Harry was 47 years old.
  Fact 0.46: Anderson, Mr. Harry is from New York City, New York, US.
  Fact 0.47: Anderson, Mr. Harry boarded at Southampton.
  Fact 0.48: Anderson, Mr. Harry's destination is New York City, New York, US.
  Fact 0.49: Andrews, Miss Kornelia Theodosia was 62 years old.
  Fact 0.50: Andrews, Miss Kornelia Theodosia is from Hudson, New York, US.
  Fact 0.51: Andrews, Miss Kornelia Theodosia boarded at Cherbourg.
  Fact 0.52: Andrews, Miss Kornelia Theodosia's destination is Hudson, New York, US.
  Fact 0.53: Andrews, Miss Kornelia Theodosia has 10 in the lifeboat or body column.
  Fact 0.54: Andrews, Mr. Thomas was 39 years old.
  Fact 0.55: Andrews, Mr. Thomas is from Belfast, Ireland, UK.
  Fact 0.56: Andrews, Mr. Thomas boarded at Belfast.
  Fact 0.57: Andrews, Mr. Thomas's destination is New York City.
  Fact 0.58: Appleton, Mrs. Charlotte Lane (née Lamson) was 53 years old.
  Fact 0.59: Appleton, Mrs. Charlotte Lane (née Lamson) is from New York City.
  Fact 0.60: Appleton, Mrs. Charlotte Lane (née Lamson) boarded at Southampton.
  Fact 0.61: Appleton, Mrs. Charlotte Lane (née Lamson)'s destination is New York City.
  Fact 0.62: Artagaveytia, Mr. Ramon was 71 years old.
  Fact 0.63: Artagaveytia, Mr. Ramon is from Buenos Aires, Argentina.
  Fact 0.64: Artagaveytia, Mr. Ramon boarded at Cherbourg.
  Fact 0.65: Artagaveytia, Mr. Ramon's destination is New York City.
  Fact 0.66: Artagaveytia, Mr. Ramon has 22 in the lifeboat or body column.
  Fact 0.67: Astor, Colonel John Jacob IV was 47 years old.
  Fact 0.68: Astor, Colonel John Jacob IV is from New York City.
  Fact 0.69: Astor, Colonel John Jacob IV boarded at Cherbourg.
  Fact 0.70: Astor, Colonel John Jacob IV's destination is New York City.
  Fact 0.71: Astor, Colonel John Jacob IV has 124 in the lifeboat or body column.
  Fact 0.72: Astor, Mrs. Madeleine Talmage (née Force) was 18 years old.
  Fact 0.73: Astor, Mrs. Madeleine Talmage (née Force) is from New York City.
  Fact 0.74: Astor, Mrs. Madeleine Talmage (née Force) boarded at Cherbourg.
  Fact 0.75: Astor, Mrs. Madeleine Talmage (née Force)'s destination is New York City.
  Fact 0.76: Miss Rosalie Bidois was 46 years old.
  Fact 0.77: Miss Rosalie Bidois is listed as the maid.
  Fact 0.78: Miss Rosalie Bidois is from New York City.
  Fact 0.79: Miss Rosalie Bidois boarded at Cherbourg.
  Fact 0.80: Miss Rosalie Bidois's destination is New York City.
  Fact 0.81: Miss Caroline Louise Endres was 39 years old.
  Fact 0.82: Miss Caroline Louise Endres is listed as the nurse.
  Fact 0.83: Miss Caroline Louise Endres is from New York City.
  Fact 0.84: Miss Caroline Louise Endres boarded at Cherbourg.
  Fact 0.85: Miss Caroline Louise Endres's destination is New York City.
  Fact 0.86: Aubart, Mrs. Léontine Pauline was 24 years old.
  Fact 0.87: Aubart, Mrs. Léontine Pauline is from Paris, France.
  Fact 0.88: Aubart, Mrs. Léontine Pauline boarded at Cherbourg.
  Fact 0.89: Aubart, Mrs. Léontine Pauline's destination is New York City.
  Fact 0.90: Aubart, Mrs. Léontine Pauline has 9 in the lifeboat or body column.
  Fact 0.91: Miss Emma Sägesser was 24 years old.
  Fact 0.92: Miss Emma Sägesser is listed as the maid.
  Fact 0.93: Miss Emma Sägesser is from Paris, France.
  Fact 0.94: Miss Emma Sägesser boarded at Cherbourg.
  Fact 0.95: Miss Emma Sägesser's destination is New York City.
  Fact 0.96: Miss Emma Sägesser has 9 in the lifeboat or body column.
  Fact 0.97: Barkworth, Mr. Algernon Henry Wilson was 47 years old.
  Fact 0.98: Barkworth, Mr. Algernon Henry Wilson is from Hessle, East Yorkshire, England, UK.
  Fact 0.99: Barkworth, Mr. Algernon Henry Wilson boarded at Southampton.
  Fact 0.100: Barkworth, Mr. Algernon Henry Wilson's destination is New York City.
  Fact 0.101: Baumann, Mr. John D. was 60 years old.
  Fact 0.102: Baumann, Mr. John D. is from New York City.
  Fact 0.103: Baumann, Mr. John D. boarded at Cherbourg.
  Fact 0.104: Baumann, Mr. John D.'s destination is New York City.
  Fact 0.105: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) was 50 years old.
  Fact 0.106: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) is from Montreal, Quebec, Canada.
  Fact 0.107: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) boarded at Cherbourg.
  Fact 0.108: Baxter, Mrs. Hélène (née de Lanaudière-Chaput)'s destination is Montreal, Quebec, Canada.
  Fact 0.109: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) has 6 in the lifeboat or body column.
  Fact 0.110: Baxter, Mr. Quigg Edmond was 24 years old.
  Fact 0.111: Baxter, Mr. Quigg Edmond is from Montreal, Quebec, Canada.
  Fact 0.112: Baxter, Mr. Quigg Edmond boarded at Cherbourg.
  Fact 0.113: Baxter, Mr. Quigg Edmond's destination is Montreal, Quebec, Canada.
  Fact 0.114: Beattie, Mr. Thomson was 36 years old.
  Fact 0.115: Beattie, Mr. Thomson is from Fergus, Ontario, Canada.
  Fact 0.116: Beattie, Mr. Thomson boarded at Southampton.
  Fact 0.117: Beattie, Mr. Thomson's destination is Fergus, Ontario, Canada.
  Fact 0.118: Beattie, Mr. Thomson has 331 in the lifeboat or body column.
  Fact 0.119: Beckwith, Mr. Richard Leonard was 37 years old.
  Fact 0.120: Beckwith, Mr. Richard Leonard is from New York City.
  Fact 0.121: Beckwith, Mr. Richard Leonard boarded at Southampton.
  Fact 0.122: Beckwith, Mr. Richard Leonard's destination is New York City.
  Fact 0.123: Beckwith, Mr. Richard Leonard has 5 in the lifeboat or body column.
  Fact 0.124: Beckwith, Mrs. Sallie (née Monypeny) was 46 years old.
  Fact 0.125: Beckwith, Mrs. Sallie (née Monypeny) is from New York City.
  Fact 0.126: Beckwith, Mrs. Sallie (née Monypeny) boarded at Southampton.
  Fact 0.127: Beckwith, Mrs. Sallie (née Monypeny)'s destination is New York City.
  Fact 0.128: Beckwith, Mrs. Sallie (née Monypeny) has 5 in the lifeboat or body column.
  Fact 0.129: Behr, Mr. Karl Howell was 26 years old.
  Fact 0.130: Behr, Mr. Karl Howell is from New York City.
  Fact 0.131: Behr, Mr. Karl Howell boarded at Cherbourg.
  Fact 0.132: Behr, Mr. Karl Howell's destination is New York City.
  Fact 0.133: Behr, Mr. Karl Howell has 5 in the lifeboat or body column.
  Fact 0.134: Birnbaum, Mr. Jakob was 24 years old.
  Fact 0.135: Birnbaum, Mr. Jakob is from Antwerp, Belgium.
  Fact 0.136: Birnbaum, Mr. Jakob boarded at Cherbourg.
  Fact 0.137: Birnbaum, Mr. Jakob's destination is San Francisco, California, US.
  Fact 0.138: Birnbaum, Mr. Jakob has 148 in the lifeboat or body column.
  Fact 0.139: Bishop, Mr. Dickinson H. "Dick" was 25 years old.
  Fact 0.140: Bishop, Mr. Dickinson H. "Dick" is from Dowagiac, Michigan, US.
  Fact 0.141: Bishop, Mr. Dickinson H. "Dick" boarded at Cherbourg.
  Fact 0.142: Bishop, Mr. Dickinson H. "Dick"'s destination is Dowagiac, Michigan, US.
  Fact 0.143: Bishop, Mr. Dickinson H. "Dick" has 7 in the lifeboat or body column.
  Fact 0.144: Bishop, Mrs. Helen (née Walton) was 19 years old.
  Fact 0.145: Bishop, Mrs. Helen (née Walton) is from Dowagiac, Michigan, US.
  Fact 0.146: Bishop, Mrs. Helen (née Walton) boarded at Cherbourg.
  Fact 0.147: Bishop, Mrs. Helen (née Walton)'s destination is Dowagiac, Michigan, US.
  Fact 0.148: Bishop, Mrs. Helen (née Walton) has 7 in the lifeboat or body column.
  Fact 0.149: Björnström-Steffanson, Mr. Mauritz Håkan was 28 years old.
  Fact 0.150: Björnström-Steffanson, Mr. Mauritz Håkan is from Stockholm, Sweden.
  Fact 0.151: Björnström-Steffanson, Mr. Mauritz Håkan boarded at Southampton.
  Fact 0.152: Björnström-Steffanson, Mr. Mauritz Håkan's destination is Washington, D.C., US.
  Fact 0.153: Blackwell, Mr. Stephen Weart was 45 years old.
  Fact 0.154: Blackwell, Mr. Stephen Weart is from Trenton, New Jersey, US.
  Fact 0.155: Blackwell, Mr. Stephen Weart boarded at Southampton.
  Fact 0.156: Blackwell, Mr. Stephen Weart's destination is Trenton, New Jersey, US.
  Fact 0.157: Blank, Mr. Henry was 39 years old.
  Fact 0.158: Blank, Mr. Henry is from Glen Ridge, New Jersey, US.
  Fact 0.159: Blank, Mr. Henry boarded at Cherbourg.
  Fact 0.160: Blank, Mr. Henry's destination is Glen Ridge, New Jersey, US.
  Fact 0.161: Blank, Mr. Henry has 7 in the lifeboat or body column.
  Fact 0.162: Bonnell, Miss Elizabeth was 61 years old.
  Fact 0.163: Bonnell, Miss Elizabeth is from Youngstown, Ohio, US.
  Fact 0.164: Bonnell, Miss Elizabeth boarded at Southampton.
  Fact 0.165: Bonnell, Miss Elizabeth's destination is Youngstown, Ohio, US.
  Fact 0.166: Bonnell, Miss Elizabeth has 8 in the lifeboat or body column.
  Fact 0.167: Bonnell, Miss Caroline was 30 years old.
  Fact 0.168: Bonnell, Miss Caroline is from Youngstown, Ohio, US.
  Fact 0.169: Bonnell, Miss Caroline boarded at Southampton.
  Fact 0.170: Bonnell, Miss Caroline's destination is Youngstown, Ohio, US.
  Fact 0.171: Bonnell, Miss Caroline has 8 in the lifeboat or body column.
  Fact 0.172: Borebank, Mr. John James was 42 years old.
  Fact 0.173: Borebank, Mr. John James is from London, England, UK.
  Fact 0.174: Borebank, Mr. John James boarded at Southampton.
  Fact 0.175: Borebank, Mr. John James's destination is Toronto, Ontario, Canada.
  Fact 0.176: Bowerman, Miss Elsie Edith was 22 years old.
  Fact 0.177: Bowerman, Miss Elsie Edith is from St. Leonards-on-Sea, East Sussex, England, UK.
  Fact 0.178: Bowerman, Miss Elsie Edith boarded at Southampton.
  Fact 0.179: Bowerman, Miss Elsie Edith's destination is New York City.
  Fact 0.180: Bowerman, Miss Elsie Edith has 6 in the lifeboat or body column.
  Fact 0.181: Brady, Mr. John Bertram was 41 years old.
  Fact 0.182: Brady, Mr. John Bertram is from Pomeroy, Washington, US.
  Fact 0.183: Brady, Mr. John Bertram boarded at Southampton.
  Fact 0.184: Brady, Mr. John Bertram's destination is Pomeroy, Washington, US.
  Fact 0.185: Brandeis, Mr. Emil was 48 years old.
  Fact 0.186: Brandeis, Mr. Emil is from Omaha, Nebraska, US.
  Fact 0.187: Brandeis, Mr. Emil boarded at Cherbourg.
  Fact 0.188: Brandeis, Mr. Emil's destination is Omaha, Nebraska, US.
  Fact 0.189: Brandeis, Mr. Emil has 208 in the lifeboat or body column.
  Fact 0.190: Brereton, Mr. George Andrew (alias George A. Brayton) was 37 years old.
  Fact 0.191: Brereton, Mr. George Andrew (alias George A. Brayton) is from Los Angeles, US.
  Fact 0.192: Brereton, Mr. George Andrew (alias George A. Brayton) boarded at Southampton.
  Fact 0.193: Brereton, Mr. George Andrew (alias George A. Brayton)'s destination is Los Angeles, California, US.
  Fact 0.194: Brereton, Mr. George Andrew (alias George A. Brayton) has 9 in the lifeboat or body column.
  Fact 0.195: Brewe, Dr. Arthur Jackson was 45 years old.
  Fact 0.196: Brewe, Dr. Arthur Jackson is from Philadelphia, Pennsylvania, US.
  Fact 0.197: Brewe, Dr. Arthur Jackson boarded at Cherbourg.
  Fact 0.198: Brewe, Dr. Arthur Jackson's destination is Philadelphia, Pennsylvania, US.
  Fact 0.199: Brown, Mrs. Caroline Lane (née Lamson) was 59 years old.
  Fact 0.200: Brown, Mrs. Caroline Lane (née Lamson) is from Belmont, Massachusetts, US.
  Fact 0.201: Brown, Mrs. Caroline Lane (née Lamson) boarded at Southampton.
  Fact 0.202: Brown, Mrs. Caroline Lane (née Lamson)'s destination is Belmont, Massachusetts, US.
  Fact 0.203: Brown, Mrs. Margaret (née Tobin) was 44 years old.
  Fact 0.204: Brown, Mrs. Margaret (née Tobin) is from Denver, Colorado, US.
  Fact 0.205: Brown, Mrs. Margaret (née Tobin) boarded at Cherbourg.
  Fact 0.206: Brown, Mrs. Margaret (née Tobin)'s destination is Denver, Colorado, US.
  Fact 0.207: Brown, Mrs. Margaret (née Tobin) has 6 in the lifeboat or body column.
  Fact 0.208: Bucknell, Mrs. Emma Eliza (née Ward) was 59 years old.
  Fact 0.209: Bucknell, Mrs. Emma Eliza (née Ward) is from Philadelphia, Pennsylvania, US.
  Fact 0.210: Bucknell, Mrs. Emma Eliza (née Ward) boarded at Cherbourg.
  Fact 0.211: Bucknell, Mrs. Emma Eliza (née Ward)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.212: Bucknell, Mrs. Emma Eliza (née Ward) has 8 in the lifeboat or body column.
  Fact 0.213: Miss Albina Bazzani was 36 years old.
  Fact 0.214: Miss Albina Bazzani is listed as the maid.
  Fact 0.215: Miss Albina Bazzani is from Philadelphia, Pennsylvania, US.
  Fact 0.216: Miss Albina Bazzani boarded at Cherbourg.
  Fact 0.217: Miss Albina Bazzani's destination is Philadelphia, Pennsylvania, US.
  Fact 0.218: Miss Albina Bazzani has 8 in the lifeboat or body column.
  Fact 0.219: Butt, Major Archibald Willingham was 46 years old.
  Fact 0.220: Butt, Major Archibald Willingham is from Washington, D.C., US.
  Fact 0.221: Butt, Major Archibald Willingham boarded at Southampton.
  Fact 0.222: Butt, Major Archibald Willingham's destination is Washington, D.C., US.
  Fact 0.223: Calderhead, Mr. Edward Pennington was 42 years old.
  Fact 0.224: Calderhead, Mr. Edward Pennington is from New York City.
  Fact 0.225: Calderhead, Mr. Edward Pennington boarded at Southampton.
  Fact 0.226: Calderhead, Mr. Edward Pennington's destination is New York City.
  Fact 0.227: Calderhead, Mr. Edward Pennington has 5 in the lifeboat or body column.
  Fact 0.228: Candee, Mrs. Helen Churchill (née Hungerford) was 53 years old.
  Fact 0.229: Candee, Mrs. Helen Churchill (née Hungerford) is from Washington, D.C., US.
  Fact 0.230: Candee, Mrs. Helen Churchill (née Hungerford) boarded at Cherbourg.
  Fact 0.231: Candee, Mrs. Helen Churchill (née Hungerford)'s destination is Washington, D.C., US.
  Fact 0.232: Candee, Mrs. Helen Churchill (née Hungerford) has 6 in the lifeboat or body column.
  Fact 0.233: Cardeza, Mrs. Charlotte Wardle (née Drake) was 58 years old.
  Fact 0.234: Cardeza, Mrs. Charlotte Wardle (née Drake) is from Germantown, Pennsylvania, US.
  Fact 0.235: Cardeza, Mrs. Charlotte Wardle (née Drake) boarded at Cherbourg.
  Fact 0.236: Cardeza, Mrs. Charlotte Wardle (née Drake)'s destination is Germantown, Pennsylvania, US.
  Fact 0.237: Miss Annie Moore Ward was 38 years old.
  Fact 0.238: Miss Annie Moore Ward is listed as the maid.
  Fact 0.239: Miss Annie Moore Ward is from Germantown, Pennsylvania, US.
  Fact 0.240: Miss Annie Moore Ward boarded at Cherbourg.
  Fact 0.241: Miss Annie Moore Ward's destination is Germantown, Pennsylvania, US.
  Fact 0.242: Cardeza, Mr. Thomas Drake Martinez was 36 years old.
  Fact 0.243: Cardeza, Mr. Thomas Drake Martinez is from Germantown, Pennsylvania, US.
  Fact 0.244: Cardeza, Mr. Thomas Drake Martinez boarded at Cherbourg.
  Fact 0.245: Cardeza, Mr. Thomas Drake Martinez's destination is Germantown, Pennsylvania, US.
  Fact 0.246: Mr. Gustave J. Lesueur was 35 years old.
  Fact 0.247: Mr. Gustave J. Lesueur is listed as the valet.
  Fact 0.248: Mr. Gustave J. Lesueur is from Germantown, Pennsylvania, US.
  Fact 0.249: Mr. Gustave J. Lesueur boarded at Cherbourg.
  Fact 0.250: Mr. Gustave J. Lesueur's destination is Germantown, Pennsylvania, US.
  Fact 0.251: Carlsson, Mr. Frans Olof was 33 years old.
  Fact 0.252: Carlsson, Mr. Frans Olof is from New York City.
  Fact 0.253: Carlsson, Mr. Frans Olof boarded at Southampton.
  Fact 0.254: Carlsson, Mr. Frans Olof's destination is New York City.
  Fact 0.255: Carrau, Mr. Francisco M. was 31 years old.
  Fact 0.256: Carrau, Mr. Francisco M. is from Montevideo, Uruguay.
  Fact 0.257: Carrau, Mr. Francisco M. boarded at Southampton.
  Fact 0.258: Carrau, Mr. Francisco M.'s destination is Montevideo, Uruguay.
  Fact 0.259: Carrau, Mr. José Pedro was 17 years old.
  Fact 0.260: Carrau, Mr. José Pedro is from Montevideo, Uruguay.
  Fact 0.261: Carrau, Mr. José Pedro boarded at Southampton.
  Fact 0.262: Carrau, Mr. José Pedro's destination is Montevideo, Uruguay.
  Fact 0.263: Carter, Mr. William Ernest was 36 years old.
  Fact 0.264: Carter, Mr. William Ernest is from Bryn Mawr, Pennsylvania, US.
  Fact 0.265: Carter, Mr. William Ernest boarded at Southampton.
  Fact 0.266: Carter, Mr. William Ernest's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.267: Mr. Alexander Cairns was 28 years old.
  Fact 0.268: Mr. Alexander Cairns is listed as the valet.
  Fact 0.269: Mr. Alexander Cairns is from Bryn Mawr, Pennsylvania, US.
  Fact 0.270: Mr. Alexander Cairns boarded at Southampton.
  Fact 0.271: Mr. Alexander Cairns's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.272: Mr. Charles Augustus Aldworth was 30 years old.
  Fact 0.273: Mr. Charles Augustus Aldworth is listed as the chauffeur.
  Fact 0.274: Mr. Charles Augustus Aldworth is from Bryn Mawr, Pennsylvania, US.
  Fact 0.275: Mr. Charles Augustus Aldworth boarded at Southampton.
  Fact 0.276: Mr. Charles Augustus Aldworth's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.277: Carter, Mrs. Lucile (née Polk) was 36 years old.
  Fact 0.278: Carter, Mrs. Lucile (née Polk) is from Bryn Mawr, Pennsylvania, US.
  Fact 0.279: Carter, Mrs. Lucile (née Polk) boarded at Southampton.
  Fact 0.280: Carter, Mrs. Lucile (née Polk)'s destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.281: Miss Auguste Serreplan was 30 years old.
  Fact 0.282: Miss Auguste Serreplan is listed as the maid.
  Fact 0.283: Miss Auguste Serreplan is from Bryn Mawr, Pennsylvania, US.
  Fact 0.284: Miss Auguste Serreplan boarded at Southampton.
  Fact 0.285: Miss Auguste Serreplan's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.286: Carter, Miss Lucile Polk was 13 years old.
  Fact 0.287: Carter, Miss Lucile Polk is from Bryn Mawr, Pennsylvania, US.
  Fact 0.288: Carter, Miss Lucile Polk boarded at Southampton.
  Fact 0.289: Carter, Miss Lucile Polk's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.290: Carter, Master William Thornton II was 11 years old.
  Fact 0.291: Carter, Master William Thornton II is from Bryn Mawr, Pennsylvania, US.
  Fact 0.292: Carter, Master William Thornton II boarded at Southampton.
  Fact 0.293: Carter, Master William Thornton II's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.294: Case, Mr. Howard Brown was 49 years old.
  Fact 0.295: Case, Mr. Howard Brown is from Ascot, Berkshire, England, UK.
  Fact 0.296: Case, Mr. Howard Brown boarded at Southampton.
  Fact 0.297: Case, Mr. Howard Brown's destination is Rochester, New York, US.
  Fact 0.298: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) was 36 years old.
  Fact 0.299: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) is from New York City.
  Fact 0.300: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) boarded at Cherbourg.
  Fact 0.301: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick)'s destination is New York City.
  Fact 0.302: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) has 5 in the lifeboat or body column.
  Fact 0.303: Cavendish, Mr. Tyrell William was 36 years old.
  Fact 0.304: Cavendish, Mr. Tyrell William is from London, England, UK.
  Fact 0.305: Cavendish, Mr. Tyrell William boarded at Southampton.
  Fact 0.306: Cavendish, Mr. Tyrell William's destination is New York City.
  Fact 0.307: Cavendish, Mr. Tyrell William has 172 in the lifeboat or body column.
  Fact 0.308: Cavendish, Mrs. Julia Florence (née Siegel) was 25 years old.
  Fact 0.309: Cavendish, Mrs. Julia Florence (née Siegel) is from London, England, UK.
  Fact 0.310: Cavendish, Mrs. Julia Florence (née Siegel) boarded at Southampton.
  Fact 0.311: Cavendish, Mrs. Julia Florence (née Siegel)'s destination is New York City.
  Fact 0.312: Cavendish, Mrs. Julia Florence (née Siegel) has 6 in the lifeboat or body column.
  Fact 0.313: Miss Ellen "Nellie" Barber was 26 years old.
  Fact 0.314: Miss Ellen "Nellie" Barber is listed as the maid.
  Fact 0.315: Miss Ellen "Nellie" Barber is from London, England, UK.
  Fact 0.316: Miss Ellen "Nellie" Barber boarded at Southampton.
  Fact 0.317: Miss Ellen "Nellie" Barber's destination is New York City.
  Fact 0.318: Miss Ellen "Nellie" Barber has 6 in the lifeboat or body column.
  Fact 0.319: Chaffee, Mr. Hubert Fuller was 46 years old.
  Fact 0.320: Chaffee, Mr. Hubert Fuller is from Amenia, North Dakota, US.
  Fact 0.321: Chaffee, Mr. Hubert Fuller boarded at Southampton.
  Fact 0.322: Chaffee, Mr. Hubert Fuller's destination is Amenia, North Dakota, US.
  Fact 0.323: Chaffee, Mrs. Carrie Constance (née Toogood) was 47 years old.
  Fact 0.324: Chaffee, Mrs. Carrie Constance (née Toogood) is from Amenia, North Dakota, US.
  Fact 0.325: Chaffee, Mrs. Carrie Constance (née Toogood) boarded at Southampton.
  Fact 0.326: Chaffee, Mrs. Carrie Constance (née Toogood)'s destination is Amenia, North Dakota, US.
  Fact 0.327: Chambers, Mr. Norman Campbell was 27 years old.
  Fact 0.328: Chambers, Mr. Norman Campbell is from New York City.
  Fact 0.329: Chambers, Mr. Norman Campbell boarded at Southampton.
  Fact 0.330: Chambers, Mr. Norman Campbell's destination is New York City.
  Fact 0.331: Chambers, Mr. Norman Campbell has 5 in the lifeboat or body column.
  Fact 0.332: Chambers, Mrs. Bertha (née Griggs) was 32 years old.
  Fact 0.333: Chambers, Mrs. Bertha (née Griggs) is from New York City.
  Fact 0.334: Chambers, Mrs. Bertha (née Griggs) boarded at Southampton.
  Fact 0.335: Chambers, Mrs. Bertha (née Griggs)'s destination is New York City.
  Fact 0.336: Chambers, Mrs. Bertha (née Griggs) has 5 in the lifeboat or body column.
  Fact 0.337: Cherry, Miss Gladys was 30 years old.
  Fact 0.338: Cherry, Miss Gladys is from London, England, UK.
  Fact 0.339: Cherry, Miss Gladys boarded at Southampton.
  Fact 0.340: Cherry, Miss Gladys's destination is Vancouver, British Columbia, Canada.
  Fact 0.341: Cherry, Miss Gladys has 8 in the lifeboat or body column.
  Fact 0.342: Chevré, Mr. Paul Romaine Marie Léonce was 45 years old.
  Fact 0.343: Chevré, Mr. Paul Romaine Marie Léonce is from Paris, France.
  Fact 0.344: Chevré, Mr. Paul Romaine Marie Léonce boarded at Cherbourg.
  Fact 0.345: Chevré, Mr. Paul Romaine Marie Léonce's destination is Ottawa, Ontario, Canada.
  Fact 0.346: Chevré, Mr. Paul Romaine Marie Léonce has 7 in the lifeboat or body column.
  Fact 0.347: Chibnall, Mrs. Edith Martha Bowerman (née Barber) was 48 years old.
  Fact 0.348: Chibnall, Mrs. Edith Martha Bowerman (née Barber) is from St. Leonards-on-Sea, East Sussex, England, UK.
  Fact 0.349: Chibnall, Mrs. Edith Martha Bowerman (née Barber) boarded at Southampton.
  Fact 0.350: Chibnall, Mrs. Edith Martha Bowerman (née Barber)'s destination is New York City.
  Fact 0.351: Chibnall, Mrs. Edith Martha Bowerman (née Barber) has 6 in the lifeboat or body column.
  Fact 0.352: Chisholm, Mr. Roderick Robert Crispin was 40 years old.
  Fact 0.353: Chisholm, Mr. Roderick Robert Crispin is from Belfast, Ireland, UK.
  Fact 0.354: Chisholm, Mr. Roderick Robert Crispin boarded at Belfast.
  Fact 0.355: Chisholm, Mr. Roderick Robert Crispin's destination is New York City.
  Fact 0.356: Clark, Mr. Walter Miller was 27 years old.
  Fact 0.357: Clark, Mr. Walter Miller is from Los Angeles, California, US.
  Fact 0.358: Clark, Mr. Walter Miller boarded at Cherbourg.
  Fact 0.359: Clark, Mr. Walter Miller's destination is Los Angeles, California, US.
  Fact 0.360: Clark, Mrs. Virginia Estelle (née McDowell) was 26 years old.
  Fact 0.361: Clark, Mrs. Virginia Estelle (née McDowell) is from Los Angeles, California, US.
  Fact 0.362: Clark, Mrs. Virginia Estelle (née McDowell) boarded at Cherbourg.
  Fact 0.363: Clark, Mrs. Virginia Estelle (née McDowell)'s destination is Los Angeles, California, US.
  Fact 0.364: Clifford, Mr. George Quincy was 40 years old.
  Fact 0.365: Clifford, Mr. George Quincy is from Stoughton, Massachusetts, US.
  Fact 0.366: Clifford, Mr. George Quincy boarded at Southampton.
  Fact 0.367: Clifford, Mr. George Quincy's destination is Stoughton, Massachusetts.
  Fact 0.368: Colley, Mr. Edward Pomeroy was 37 years old.
  Fact 0.369: Colley, Mr. Edward Pomeroy is from Dublin, Ireland, UK.
  Fact 0.370: Colley, Mr. Edward Pomeroy boarded at Southampton.
  Fact 0.371: Colley, Mr. Edward Pomeroy's destination is Vancouver, British Columbia, Canada.
  Fact 0.372: Compton, Mrs. Mary Eliza (née Ingersoll) was 64 years old.
  Fact 0.373: Compton, Mrs. Mary Eliza (née Ingersoll) is from Lakewood, New Jersey, US.
  Fact 0.374: Compton, Mrs. Mary Eliza (née Ingersoll) boarded at Cherbourg.
  Fact 0.375: Compton, Mrs. Mary Eliza (née Ingersoll)'s destination is Lakewood, New Jersey, US.
  Fact 0.376: Compton, Mrs. Mary Eliza (née Ingersoll) has 14 in the lifeboat or body column.
  Fact 0.377: Compton, Miss Sara Rebecca was 39 years old.
  Fact 0.378: Compton, Miss Sara Rebecca is from Lakewood, New Jersey, US.
  Fact 0.379: Compton, Miss Sara Rebecca boarded at Cherbourg.
  Fact 0.380: Compton, Miss Sara Rebecca's destination is Lakewood, New Jersey, US.
  Fact 0.381: Compton, Miss Sara Rebecca has 14 in the lifeboat or body column.
  Fact 0.382: Compton, Mr. Alexander Taylor Jr. was 37 years old.
  Fact 0.383: Compton, Mr. Alexander Taylor Jr. is from Lakewood, New Jersey, US.
  Fact 0.384: Compton, Mr. Alexander Taylor Jr. boarded at Cherbourg.
  Fact 0.385: Compton, Mr. Alexander Taylor Jr.'s destination is Lakewood, New Jersey, US.
  Fact 0.386: Cornell, Mrs. Malvina Helen (née Lamson) was 55 years old.
  Fact 0.387: Cornell, Mrs. Malvina Helen (née Lamson) is from New York City.
  Fact 0.388: Cornell, Mrs. Malvina Helen (née Lamson) boarded at Southampton.
  Fact 0.389: Cornell, Mrs. Malvina Helen (née Lamson)'s destination is New York City.
  Fact 0.390: Crafton, Mr. John Bertram was 59 years old.
  Fact 0.391: Crafton, Mr. John Bertram is from Roachdale, Indiana, US.
  Fact 0.392: Crafton, Mr. John Bertram boarded at Southampton.
  Fact 0.393: Crafton, Mr. John Bertram's destination is Roachdale, Indiana, US.
  Fact 0.394: Crosby, Captain Edward was 70 years old.
  Fact 0.395: Crosby, Captain Edward is from Milwaukee, Wisconsin, US.
  Fact 0.396: Crosby, Captain Edward boarded at Southampton.
  Fact 0.397: Crosby, Captain Edward's destination is Milwaukee, Wisconsin, US.
  Fact 0.398: Crosby, Captain Edward has 269 in the lifeboat or body column.
  Fact 0.399: Crosby, Mrs. Catherine Elizabeth (née Halstead) was 64 years old.
  Fact 0.400: Crosby, Mrs. Catherine Elizabeth (née Halstead) is from Milwaukee, Wisconsin, US.
  Fact 0.401: Crosby, Mrs. Catherine Elizabeth (née Halstead) boarded at Southampton.
  Fact 0.402: Crosby, Mrs. Catherine Elizabeth (née Halstead)'s destination is Milwaukee, Wisconsin, US.
  Fact 0.403: Crosby, Mrs. Catherine Elizabeth (née Halstead) has 7 in the lifeboat or body column.
  Fact 0.404: Crosby, Miss Harriette Rebecca was 39 years old.
  Fact 0.405: Crosby, Miss Harriette Rebecca is from Milwaukee, Wisconsin, US.
  Fact 0.406: Crosby, Miss Harriette Rebecca boarded at Southampton.
  Fact 0.407: Crosby, Miss Harriette Rebecca's destination is Milwaukee, Wisconsin, US.
  Fact 0.408: Crosby, Miss Harriette Rebecca has 7 in the lifeboat or body column.
  Fact 0.409: Cumings, Mr. John Bradley was 39 years old.
  Fact 0.410: Cumings, Mr. John Bradley is from New York City.
  Fact 0.411: Cumings, Mr. John Bradley boarded at Cherbourg.
  Fact 0.412: Cumings, Mr. John Bradley's destination is New York City.
  Fact 0.413: Cumings, Mrs. Florence Briggs (née Thayer) was 35 years old.
  Fact 0.414: Cumings, Mrs. Florence Briggs (née Thayer) is from New York City.
  Fact 0.415: Cumings, Mrs. Florence Briggs (née Thayer) boarded at Cherbourg.
  Fact 0.416: Cumings, Mrs. Florence Briggs (née Thayer)'s destination is New York, US.
  Fact 0.417: Daly, Mr. Peter Dennis was 51 years old.
  Fact 0.418: Daly, Mr. Peter Dennis is from Lima, Peru.
  Fact 0.419: Daly, Mr. Peter Dennis boarded at Southampton.
  Fact 0.420: Daly, Mr. Peter Dennis's destination is Lima, Peru.
  Fact 0.421: Daniel, Mr. Robert Williams was 27 years old.
  Fact 0.422: Daniel, Mr. Robert Williams is from Philadelphia, Pennsylvania, US.
  Fact 0.423: Daniel, Mr. Robert Williams boarded at Southampton.
  Fact 0.424: Daniel, Mr. Robert Williams's destination is Philadelphia, Pennsylvania, US.
  Fact 0.425: Davidson, Mr. Thornton was 31 years old.
  Fact 0.426: Davidson, Mr. Thornton is from Montreal, Quebec, Canada.
  Fact 0.427: Davidson, Mr. Thornton boarded at Southampton.
  Fact 0.428: Davidson, Mr. Thornton's destination is Montreal, Quebec, Canada.
  Fact 0.429: Davidson, Mrs. Orian (née Hays) was 27 years old.
  Fact 0.430: Davidson, Mrs. Orian (née Hays) is from Montreal, Quebec, Canada.
  Fact 0.431: Davidson, Mrs. Orian (née Hays) boarded at Southampton.
  Fact 0.432: Davidson, Mrs. Orian (née Hays)'s destination is Montreal, Quebec, Canada.
  Fact 0.433: Dick, Mr. Albert Adrian was 31 years old.
  Fact 0.434: Dick, Mr. Albert Adrian is from Calgary, Alberta, Canada.
  Fact 0.435: Dick, Mr. Albert Adrian boarded at Southampton.
  Fact 0.436: Dick, Mr. Albert Adrian's destination is Calgary, Alberta, Canada.
  Fact 0.437: Dick, Mrs. Vera (née Gillespie) was 17 years old.
  Fact 0.438: Dick, Mrs. Vera (née Gillespie) is from Calgary, Alberta, Canada.
  Fact 0.439: Dick, Mrs. Vera (née Gillespie) boarded at Southampton.
  Fact 0.440: Dick, Mrs. Vera (née Gillespie)'s destination is Calgary, Alberta, Canada.
  Fact 0.441: Dineston, Mrs. Nancy May Bowles was 28 years old.
  Fact 0.442: Dineston, Mrs. Nancy May Bowles is from London, England, UK.
  Fact 0.443: Dineston, Mrs. Nancy May Bowles boarded at Southampton.
  Fact 0.444: Dineston, Mrs. Nancy May Bowles's destination is New York City, USA.
  Fact 0.445: Dineston, George Samuel was 30 years old.
  Fact 0.446: Dineston, George Samuel is from London, England, UK.
  Fact 0.447: Dineston, George Samuel boarded at Southampton.
  Fact 0.448: Dineston, George Samuel's destination is New York City, USA.
  Fact 0.449: Dodge, Dr. Washington was 52 years old.
  Fact 0.450: Dodge, Dr. Washington is from San Francisco, California, US.
  Fact 0.451: Dodge, Dr. Washington boarded at Southampton.
  Fact 0.452: Dodge, Dr. Washington's destination is San Francisco, California, US.
  Fact 0.453: Dodge, Dr. Washington has 13 in the lifeboat or body column.
  Fact 0.454: Dodge, Mrs. Ruth (née Vidaver) was 34 years old.
  Fact 0.455: Dodge, Mrs. Ruth (née Vidaver) is from San Francisco, California, US.
  Fact 0.456: Dodge, Mrs. Ruth (née Vidaver) boarded at Southampton.
  Fact 0.457: Dodge, Mrs. Ruth (née Vidaver)'s destination is San Francisco, California, US.
  Fact 0.458: Dodge, Mrs. Ruth (née Vidaver) has 5 in the lifeboat or body column.
  Fact 0.459: Dodge, Master Washington, Jr. is from San Francisco, California, US.
  Fact 0.460: Dodge, Master Washington, Jr. boarded at Southampton.
  Fact 0.461: Dodge, Master Washington, Jr.'s destination is San Francisco, California, US.
  Fact 0.462: Dodge, Master Washington, Jr. has 5 in the lifeboat or body column.
  Fact 0.463: Douglas, Mr. Walter Donald was 50 years old.
  Fact 0.464: Douglas, Mr. Walter Donald is from Minneapolis, Minnesota, US.
  Fact 0.465: Douglas, Mr. Walter Donald boarded at Cherbourg.
  Fact 0.466: Douglas, Mr. Walter Donald's destination is Minneapolis, Minnesota, US.
  Fact 0.467: Douglas, Mr. Walter Donald has 62 in the lifeboat or body column.
  Fact 0.468: Douglas, Mrs. Mahala (née Dutton) was 48 years old.
  Fact 0.469: Douglas, Mrs. Mahala (née Dutton) is from Minneapolis, Minnesota, US.
  Fact 0.470: Douglas, Mrs. Mahala (née Dutton) boarded at Cherbourg.
  Fact 0.471: Douglas, Mrs. Mahala (née Dutton)'s destination is Minneapolis, Minnesota, US.
  Fact 0.472: Miss Berthe Leroy was 27 years old.
  Fact 0.473: Miss Berthe Leroy is listed as the maid.
  Fact 0.474: Miss Berthe Leroy is from Minneapolis, Minnesota, US.
  Fact 0.475: Miss Berthe Leroy boarded at Cherbourg.
  Fact 0.476: Miss Berthe Leroy's destination is Minneapolis, Minnesota, US.
  Fact 0.477: Douglas, Mrs. Mary Hélène (née Baxter) was 27 years old.
  Fact 0.478: Douglas, Mrs. Mary Hélène (née Baxter) is from Montreal, Quebec, Canada.
  Fact 0.479: Douglas, Mrs. Mary Hélène (née Baxter) boarded at Cherbourg.
  Fact 0.480: Douglas, Mrs. Mary Hélène (née Baxter)'s destination is Montreal, Quebec, Canada.
  Fact 0.481: Douglas, Mrs. Mary Hélène (née Baxter) has 6 in the lifeboat or body column.
  Fact 0.482: Duff Gordon, Sir Cosmo Edmund was 49 years old.
  Fact 0.483: Duff Gordon, Sir Cosmo Edmund is from London, England, UK.
  Fact 0.484: Duff Gordon, Sir Cosmo Edmund boarded at Cherbourg.
  Fact 0.485: Duff Gordon, Sir Cosmo Edmund's destination is New York City.
  Fact 0.486: Duff Gordon, Lucy Christiana, Lady (née Sutherland) was 48 years old.
  Fact 0.487: Duff Gordon, Lucy Christiana, Lady (née Sutherland) is from London, England, UK.
  Fact 0.488: Duff Gordon, Lucy Christiana, Lady (née Sutherland) boarded at Cherbourg.
  Fact 0.489: Duff Gordon, Lucy Christiana, Lady (née Sutherland)'s destination is New York City.
  Fact 0.490: Miss Laura Mabel Francatelli was 31 years old.
  Fact 0.491: Miss Laura Mabel Francatelli is listed as the secretary.
  Fact 0.492: Miss Laura Mabel Francatelli is from London, England, UK.
  Fact 0.493: Miss Laura Mabel Francatelli boarded at Cherbourg.
  Fact 0.494: Miss Laura Mabel Francatelli's destination is New York City.
  Fact 0.495: Dulles, Mr. William Crothers was 39 years old.
  Fact 0.496: Dulles, Mr. William Crothers is from Philadelphia, Pennsylvania, US.
  Fact 0.497: Dulles, Mr. William Crothers boarded at Cherbourg.
  Fact 0.498: Dulles, Mr. William Crothers's destination is Philadelphia, Pennsylvania, US.
  Fact 0.499: Dulles, Mr. William Crothers has 133 in the lifeboat or body column.
  Fact 0.500: Eustis, Miss Elizabeth Mussey was 54 years old.
  Fact 0.501: Eustis, Miss Elizabeth Mussey is from Brookline, Massachusetts, US.
  Fact 0.502: Eustis, Miss Elizabeth Mussey boarded at Cherbourg.
  Fact 0.503: Eustis, Miss Elizabeth Mussey's destination is Brookline, Massachusetts, US.
  Fact 0.504: Evans, Miss Edith Corse was 36 years old.
  Fact 0.505: Evans, Miss Edith Corse is from New York City.
  Fact 0.506: Evans, Miss Edith Corse boarded at Cherbourg.
  Fact 0.507: Evans, Miss Edith Corse's destination is New York City.
  Fact 0.508: Flegenheim, Mrs. Antoinette (née Wendt) was 48 years old.
  Fact 0.509: Flegenheim, Mrs. Antoinette (née Wendt) is from New York City.
  Fact 0.510: Flegenheim, Mrs. Antoinette (née Wendt) boarded at Cherbourg.
  Fact 0.511: Flegenheim, Mrs. Antoinette (née Wendt)'s destination is New York City.
  Fact 0.512: Flegenheim, Mrs. Antoinette (née Wendt) has 7 in the lifeboat or body column.
  Fact 0.513: Flynn, Mr. John Irwin was 36 years old.
  Fact 0.514: Flynn, Mr. John Irwin is from Brooklyn, New York, US.
  Fact 0.515: Flynn, Mr. John Irwin boarded at Southampton.
  Fact 0.516: Flynn, Mr. John Irwin's destination is Brooklyn, New York, US.
  Fact 0.517: Flynn, Mr. John Irwin has 5 in the lifeboat or body column.
  Fact 0.518: Foreman, Mr. Benjamin Laventall was 30 years old.
  Fact 0.519: Foreman, Mr. Benjamin Laventall is from New York City.
  Fact 0.520: Foreman, Mr. Benjamin Laventall boarded at Southampton.
  Fact 0.521: Foreman, Mr. Benjamin Laventall's destination is New York City.
  Fact 0.522: Fortune, Mr. Mark was 64 years old.
  Fact 0.523: Fortune, Mr. Mark is from Winnipeg, Manitoba, Canada.
  Fact 0.524: Fortune, Mr. Mark boarded at Southampton.
  Fact 0.525: Fortune, Mr. Mark's destination is Winnipeg, Manitoba, Canada.
  Fact 0.526: Fortune, Mrs. Mary (née McDougald) was 60 years old.
  Fact 0.527: Fortune, Mrs. Mary (née McDougald) is from Winnipeg, Manitoba, Canada.
  Fact 0.528: Fortune, Mrs. Mary (née McDougald) boarded at Southampton.
  Fact 0.529: Fortune, Mrs. Mary (née McDougald)'s destination is Winnipeg, Manitoba, Canada.
  Fact 0.530: Fortune, Mrs. Mary (née McDougald) has 10 in the lifeboat or body column.
  Fact 0.531: Fortune, Miss Ethel Flora was 28 years old.
  Fact 0.532: Fortune, Miss Ethel Flora is from Winnipeg, Manitoba, Canada.
  Fact 0.533: Fortune, Miss Ethel Flora boarded at Southampton.
  Fact 0.534: Fortune, Miss Ethel Flora's destination is Winnipeg, Manitoba, Canada.
  Fact 0.535: Fortune, Miss Ethel Flora has 10 in the lifeboat or body column.
  Fact 0.536: Fortune, Miss Alice Elizabeth was 24 years old.
  Fact 0.537: Fortune, Miss Alice Elizabeth is from Winnipeg, Manitoba, Canada.
  Fact 0.538: Fortune, Miss Alice Elizabeth boarded at Southampton.
  Fact 0.539: Fortune, Miss Alice Elizabeth's destination is Winnipeg, Manitoba, Canada.
  Fact 0.540: Fortune, Miss Alice Elizabeth has 10 in the lifeboat or body column.
  Fact 0.541: Fortune, Miss Mabel Helen was 23 years old.
  Fact 0.542: Fortune, Miss Mabel Helen is from Winnipeg, Manitoba, Canada.
  Fact 0.543: Fortune, Miss Mabel Helen boarded at Southampton.
  Fact 0.544: Fortune, Miss Mabel Helen's destination is Winnipeg, Manitoba, Canada.
  Fact 0.545: Fortune, Miss Mabel Helen has 10 in the lifeboat or body column.
  Fact 0.546: Fortune, Mr. Charles Alexander was 19 years old.
  Fact 0.547: Fortune, Mr. Charles Alexander is from Winnipeg, Manitoba, Canada.
  Fact 0.548: Fortune, Mr. Charles Alexander boarded at Southampton.
  Fact 0.549: Fortune, Mr. Charles Alexander's destination is Winnipeg, Manitoba, Canada.
  Fact 0.550: Franklin, Mr. Thomas Parnham was 37 years old.
  Fact 0.551: Franklin, Mr. Thomas Parnham is from London, England, UK.
  Fact 0.552: Franklin, Mr. Thomas Parnham boarded at Southampton.
  Fact 0.553: Franklin, Mr. Thomas Parnham's destination is New York City.
  Fact 0.554: Frauenthal, Dr. Henry William was 49 years old.
  Fact 0.555: Frauenthal, Dr. Henry William is from London, England, UK.
  Fact 0.556: Frauenthal, Dr. Henry William boarded at Southampton.
  Fact 0.557: Frauenthal, Dr. Henry William's destination is New York City.
  Fact 0.558: Frauenthal, Dr. Henry William has 5 in the lifeboat or body column.
  Fact 0.559: Frauenthal, Mrs. Clara (née Heinsheimer) was 42 years old.
  Fact 0.560: Frauenthal, Mrs. Clara (née Heinsheimer) is from London, England, UK.
  Fact 0.561: Frauenthal, Mrs. Clara (née Heinsheimer) boarded at Southampton.
  Fact 0.562: Frauenthal, Mrs. Clara (née Heinsheimer)'s destination is New York City.
  Fact 0.563: Frauenthal, Mrs. Clara (née Heinsheimer) has 5 in the lifeboat or body column.
  Fact 0.564: Frauenthal, Mr. Isaac Gerald was 43 years old.
  Fact 0.565: Frauenthal, Mr. Isaac Gerald is from London, England, UK.
  Fact 0.566: Frauenthal, Mr. Isaac Gerald boarded at Southampton.
  Fact 0.567: Frauenthal, Mr. Isaac Gerald's destination is New York City.
  Fact 0.568: Frauenthal, Mr. Isaac Gerald has 5 in the lifeboat or body column.
  Fact 0.569: Frölicher, Mr. Maximilian Josef was 60 years old.
  Fact 0.570: Frölicher, Mr. Maximilian Josef is from Zürich, Switzerland.
  Fact 0.571: Frölicher, Mr. Maximilian Josef boarded at Cherbourg.
  Fact 0.572: Frölicher, Mr. Maximilian Josef's destination is New York City.
  Fact 0.573: Frölicher, Mr. Maximilian Josef has 5 in the lifeboat or body column.
  Fact 0.574: Frölicher, Mrs. Margaretha Emerentia (née Stehli) was 48 years old.
  Fact 0.575: Frölicher, Mrs. Margaretha Emerentia (née Stehli) is from Zürich, Switzerland.
  Fact 0.576: Frölicher, Mrs. Margaretha Emerentia (née Stehli) boarded at Cherbourg.
  Fact 0.577: Frölicher, Mrs. Margaretha Emerentia (née Stehli)'s destination is New York City.
  Fact 0.578: Frölicher, Mrs. Margaretha Emerentia (née Stehli) has 5 in the lifeboat or body column.
  Fact 0.579: Frölicher-Stehli, Miss Hedwig Margaritha was 22 years old.
  Fact 0.580: Frölicher-Stehli, Miss Hedwig Margaritha is from Zürich, Switzerland.
  Fact 0.581: Frölicher-Stehli, Miss Hedwig Margaritha boarded at Cherbourg.
  Fact 0.582: Frölicher-Stehli, Miss Hedwig Margaritha's destination is New York City.
  Fact 0.583: Frölicher-Stehli, Miss Hedwig Margaritha has 5 in the lifeboat or body column.
  Fact 0.584: Futrelle, Mr. Jacques Heath was 37 years old.
  Fact 0.585: Futrelle, Mr. Jacques Heath is from Scituate, Massachusetts, US.
  Fact 0.586: Futrelle, Mr. Jacques Heath boarded at Southampton.
  Fact 0.587: Futrelle, Mr. Jacques Heath's destination is Scituate, Massachusetts, US.
  Fact 0.588: Futrelle, Mrs. Lily May (née Peel) was 35 years old.
  Fact 0.589: Futrelle, Mrs. Lily May (née Peel) is from Scituate, Massachusetts, US.
  Fact 0.590: Futrelle, Mrs. Lily May (née Peel) boarded at Southampton.
  Fact 0.591: Futrelle, Mrs. Lily May (née Peel)'s destination is Scituate, Massachusetts, US.
  Fact 0.592: Gee, Mr. Arthur H. was 47 years old.
  Fact 0.593: Gee, Mr. Arthur H. is from St. Annes-on-Sea, Lancashire, England, UK.
  Fact 0.594: Gee, Mr. Arthur H. boarded at Southampton.
  Fact 0.595: Gee, Mr. Arthur H.'s destination is Mexico City, Mexico.
  Fact 0.596: Gee, Mr. Arthur H. has 275 in the lifeboat or body column.
  Fact 0.597: Gibson, Mrs. Pauline Caroline (née Boeson) was 44 years old.
  Fact 0.598: Gibson, Mrs. Pauline Caroline (née Boeson) is from New York City.
  Fact 0.599: Gibson, Mrs. Pauline Caroline (née Boeson) boarded at Cherbourg.
  Fact 0.600: Gibson, Mrs. Pauline Caroline (née Boeson)'s destination is New York City.
  Fact 0.601: Gibson, Mrs. Pauline Caroline (née Boeson) has 7 in the lifeboat or body column.
  Fact 0.602: Gibson, Miss Dorothy Winifred was 22 years old.
  Fact 0.603: Gibson, Miss Dorothy Winifred is from New York City.
  Fact 0.604: Gibson, Miss Dorothy Winifred boarded at Cherbourg.
  Fact 0.605: Gibson, Miss Dorothy Winifred's destination is New York City.
  Fact 0.606: Gibson, Miss Dorothy Winifred has 7 in the lifeboat or body column.
  Fact 0.607: Goldenberg, Mr. Samuel L. was 47 years old.
  Fact 0.608: Goldenberg, Mr. Samuel L. is from Paris, France.
  Fact 0.609: Goldenberg, Mr. Samuel L. boarded at Cherbourg.
  Fact 0.610: Goldenberg, Mr. Samuel L.'s destination is New York City.
  Fact 0.611: Goldenberg, Mr. Samuel L. has 5 in the lifeboat or body column.
  Fact 0.612: Goldenberg, Mrs. Nella (née Wiggins) was 40 years old.
  Fact 0.613: Goldenberg, Mrs. Nella (née Wiggins) is from Paris, France.
  Fact 0.614: Goldenberg, Mrs. Nella (née Wiggins) boarded at Cherbourg.
  Fact 0.615: Goldenberg, Mrs. Nella (née Wiggins)'s destination is New York City.
  Fact 0.616: Goldenberg, Mrs. Nella (née Wiggins) has 5 in the lifeboat or body column.
  Fact 0.617: Goldschmidt, Mr. George B. was 71 years old.
  Fact 0.618: Goldschmidt, Mr. George B. is from New York City.
  Fact 0.619: Goldschmidt, Mr. George B. boarded at Cherbourg.
  Fact 0.620: Goldschmidt, Mr. George B.'s destination is New York City.
  Fact 0.621: Gracie IV, Colonel Archibald was 53 years old.
  Fact 0.622: Gracie IV, Colonel Archibald is from Washington, D.C., US.
  Fact 0.623: Gracie IV, Colonel Archibald boarded at Southampton.
  Fact 0.624: Gracie IV, Colonel Archibald's destination is Washington, D.C., US.
  Fact 0.625: Graham, Mr. George Edward was 38 years old.
  Fact 0.626: Graham, Mr. George Edward is from Winnipeg, Manitoba, Canada.
  Fact 0.627: Graham, Mr. George Edward boarded at Southampton.
  Fact 0.628: Graham, Mr. George Edward's destination is Winnipeg, Manitoba, Canada.
  Fact 0.629: Graham, Mr. George Edward has 147 in the lifeboat or body column.
  Fact 0.630: Graham, Mrs. Edith Ware (née Junkins) was 59 years old.
  Fact 0.631: Graham, Mrs. Edith Ware (née Junkins) is from Greenwich, Connecticut, US.
  Fact 0.632: Graham, Mrs. Edith Ware (née Junkins) boarded at Southampton.
  Fact 0.633: Graham, Mrs. Edith Ware (née Junkins)'s destination is Greenwich, Connecticut, US.
  Fact 0.634: Graham, Miss Margaret Edith was 19 years old.
  Fact 0.635: Graham, Miss Margaret Edith is from Greenwich, Connecticut, US.
  Fact 0.636: Graham, Miss Margaret Edith boarded at Southampton.
  Fact 0.637: Graham, Miss Margaret Edith's destination is Greenwich, Connecticut, US.
  Fact 0.638: Miss Elizabeth Weed Shutes was 40 years old.
  Fact 0.639: Miss Elizabeth Weed Shutes is listed as the governess.
  Fact 0.640: Miss Elizabeth Weed Shutes is from Greenwich, Connecticut, US.
  Fact 0.641: Miss Elizabeth Weed Shutes boarded at Southampton.
  Fact 0.642: Miss Elizabeth Weed Shutes's destination is Greenwich, Connecticut, US.
  Fact 0.643: Greenfield, Mrs. Blanche (née Strouse) was 45 years old.
  Fact 0.644: Greenfield, Mrs. Blanche (née Strouse) is from New York City.
  Fact 0.645: Greenfield, Mrs. Blanche (née Strouse) boarded at Cherbourg.
  Fact 0.646: Greenfield, Mrs. Blanche (née Strouse)'s destination is New York City.
  Fact 0.647: Greenfield, Mrs. Blanche (née Strouse) has 7 in the lifeboat or body column.
  Fact 0.648: Greenfield, Mr. William Bertram was 23 years old.
  Fact 0.649: Greenfield, Mr. William Bertram is from New York City.
  Fact 0.650: Greenfield, Mr. William Bertram boarded at Cherbourg.
  Fact 0.651: Greenfield, Mr. William Bertram's destination is New York City.
  Fact 0.652: Greenfield, Mr. William Bertram has 7 in the lifeboat or body column.
  Fact 0.653: Guggenheim, Mr. Benjamin was 46 years old.
  Fact 0.654: Guggenheim, Mr. Benjamin is from Paris, France.
  Fact 0.655: Guggenheim, Mr. Benjamin boarded at Cherbourg.
  Fact 0.656: Guggenheim, Mr. Benjamin's destination is New York City.
  Fact 0.657: Mr. Victor Giglio was 24 years old.
  Fact 0.658: Mr. Victor Giglio is listed as the valet.
  Fact 0.659: Mr. Victor Giglio is from Paris, France.
  Fact 0.660: Mr. Victor Giglio boarded at Cherbourg.
  Fact 0.661: Mr. Victor Giglio's destination is New York City.
  Fact 0.662: Mr. René Pernot was 39 years old.
  Fact 0.663: Mr. René Pernot is listed as the chauffeur.
  Fact 0.664: Mr. René Pernot is from Paris, France.
  Fact 0.665: Mr. René Pernot boarded at Cherbourg.
  Fact 0.666: Mr. René Pernot's destination is New York City.
  Fact 0.667: Harder, Mr. George Achilles was 25 years old.
  Fact 0.668: Harder, Mr. George Achilles is from New York City.
  Fact 0.669: Harder, Mr. George Achilles boarded at Cherbourg.
  Fact 0.670: Harder, Mr. George Achilles's destination is New York City.
  Fact 0.671: Harder, Mr. George Achilles has 5 in the lifeboat or body column.
  Fact 0.672: Harder, Mrs. Dorothy (née Annan) was 21 years old.
  Fact 0.673: Harder, Mrs. Dorothy (née Annan) is from New York City.
  Fact 0.674: Harder, Mrs. Dorothy (née Annan) boarded at Cherbourg.
  Fact 0.675: Harder, Mrs. Dorothy (née Annan)'s destination is New York City.
  Fact 0.676: Harder, Mrs. Dorothy (née Annan) has 5 in the lifeboat or body column.
  Fact 0.677: Harper, Mr. Henry Sleeper was 48 years old.
  Fact 0.678: Harper, Mr. Henry Sleeper is from New York City.
  Fact 0.679: Harper, Mr. Henry Sleeper boarded at Cherbourg.
  Fact 0.680: Harper, Mr. Henry Sleeper's destination is New York City.
  Fact 0.681: Mr. Hammad Hassab was 27 years old.
  Fact 0.682: Mr. Hammad Hassab is listed as the dragoman.
  Fact 0.683: Mr. Hammad Hassab is from Cairo, Egypt.
  Fact 0.684: Mr. Hammad Hassab boarded at Cherbourg.
  Fact 0.685: Mr. Hammad Hassab's destination is New York City.
  Fact 0.686: Harper, Mrs. Myna (née Haxtun) was 49 years old.
  Fact 0.687: Harper, Mrs. Myna (née Haxtun) is from New York City.
  Fact 0.688: Harper, Mrs. Myna (née Haxtun) boarded at Cherbourg.
  Fact 0.689: Harper, Mrs. Myna (née Haxtun)'s destination is New York City.
  Fact 0.690: Harris, Mr. Henry Birkhardt was 45 years old.
  Fact 0.691: Harris, Mr. Henry Birkhardt is from New York City.
  Fact 0.692: Harris, Mr. Henry Birkhardt boarded at Southampton.
  Fact 0.693: Harris, Mr. Henry Birkhardt's destination is New York City.
  Fact 0.694: Harris, Mrs. Irene (née Wallach) was 35 years old.
  Fact 0.695: Harris, Mrs. Irene (née Wallach) is from New York City.
  Fact 0.696: Harris, Mrs. Irene (née Wallach) boarded at Southampton.
  Fact 0.697: Harris, Mrs. Irene (née Wallach)'s destination is New York City.
  Fact 0.698: Hawksford, Mr. Walter James was 45 years old.
  Fact 0.699: Hawksford, Mr. Walter James is from Kingston, Surrey, England, UK.
  Fact 0.700: Hawksford, Mr. Walter James boarded at Southampton.
  Fact 0.701: Hawksford, Mr. Walter James's destination is New York City.
  Fact 0.702: Hays, Mr. Charles Melville was 55 years old.
  Fact 0.703: Hays, Mr. Charles Melville is from Montreal, Quebec, Canada.
  Fact 0.704: Hays, Mr. Charles Melville boarded at Southampton.
  Fact 0.705: Hays, Mr. Charles Melville's destination is Montreal, Quebec, Canada.
  Fact 0.706: Hays, Mr. Charles Melville has 307 in the lifeboat or body column.
  Fact 0.707: Mr. Vivian Ponsonby Payne was 22 years old.
  Fact 0.708: Mr. Vivian Ponsonby Payne is listed as the clerk.
  Fact 0.709: Mr. Vivian Ponsonby Payne is from Montreal, Quebec, Canada.
  Fact 0.710: Mr. Vivian Ponsonby Payne boarded at Southampton.
  Fact 0.711: Mr. Vivian Ponsonby Payne's destination is Montreal, Quebec, Canada.
  Fact 0.712: Hays, Mrs. Clara Jennings (née Grigg) was 52 years old.
  Fact 0.713: Hays, Mrs. Clara Jennings (née Grigg) is from Montreal, Quebec, Canada.
  Fact 0.714: Hays, Mrs. Clara Jennings (née Grigg) boarded at Southampton.
  Fact 0.715: Hays, Mrs. Clara Jennings (née Grigg)'s destination is Montreal, Quebec, Canada.
  Fact 0.716: Miss Mary Anne Perreault was 33 years old.
  Fact 0.717: Miss Mary Anne Perreault is listed as the maid.
  Fact 0.718: Miss Mary Anne Perreault is from Montreal, Quebec, Canada.
  Fact 0.719: Miss Mary Anne Perreault boarded at Southampton.
  Fact 0.720: Miss Mary Anne Perreault's destination is Montreal, Quebec, Canada.
  Fact 0.721: Hays, Miss Margaret Bechstein was 24 years old.
  Fact 0.722: Hays, Miss Margaret Bechstein is from New York City.
  Fact 0.723: Hays, Miss Margaret Bechstein boarded at Cherbourg.
  Fact 0.724: Hays, Miss Margaret Bechstein's destination is New York City.
  Fact 0.725: Hays, Miss Margaret Bechstein has 7 in the lifeboat or body column.
  Fact 0.726: Head, Mr. Christopher was 42 years old.
  Fact 0.727: Head, Mr. Christopher is from Chelsea, London, England, UK.
  Fact 0.728: Head, Mr. Christopher boarded at Southampton.
  Fact 0.729: Head, Mr. Christopher's destination is New York City.
  Fact 0.730: Hilliard, Mr. Herbert Henry was 44 years old.
  Fact 0.731: Hilliard, Mr. Herbert Henry is from Brighton, Massachusetts, US.
  Fact 0.732: Hilliard, Mr. Herbert Henry boarded at Southampton.
  Fact 0.733: Hilliard, Mr. Herbert Henry's destination is Brighton, Massachusetts, US.
  Fact 0.734: Hipkins, Mr. William Edward was 55 years old.
  Fact 0.735: Hipkins, Mr. William Edward is from Birmingham, West Midlands, England, UK.
  Fact 0.736: Hipkins, Mr. William Edward boarded at Southampton.
  Fact 0.737: Hipkins, Mr. William Edward's destination is New York City.
  Fact 0.738: Hippach, Mrs. Ida Sophia (née Fischer) was 44 years old.
  Fact 0.739: Hippach, Mrs. Ida Sophia (née Fischer) is from Chicago, Illinois, US.
  Fact 0.740: Hippach, Mrs. Ida Sophia (née Fischer) boarded at Cherbourg.
  Fact 0.741: Hippach, Mrs. Ida Sophia (née Fischer)'s destination is Chicago, Illinois, US.
  Fact 0.742: Hippach, Miss Jean Gertrude was 17 years old.
  Fact 0.743: Hippach, Miss Jean Gertrude is from Chicago, Illinois, US.
  Fact 0.744: Hippach, Miss Jean Gertrude boarded at Cherbourg.
  Fact 0.745: Hippach, Miss Jean Gertrude's destination is Chicago, Illinois, US.
  Fact 0.746: Hogeboom, Mrs. Anna Louisa (née Andrews) was 51 years old.
  Fact 0.747: Hogeboom, Mrs. Anna Louisa (née Andrews) is from Hudson, New York, US.
  Fact 0.748: Hogeboom, Mrs. Anna Louisa (née Andrews) boarded at Cherbourg.
  Fact 0.749: Hogeboom, Mrs. Anna Louisa (née Andrews)'s destination is Hudson, New York, US.
  Fact 0.750: Hogeboom, Mrs. Anna Louisa (née Andrews) has 10 in the lifeboat or body column.
  Fact 0.751: Holverson, Mr. Alexander Oskar was 42 years old.
  Fact 0.752: Holverson, Mr. Alexander Oskar is from New York City.
  Fact 0.753: Holverson, Mr. Alexander Oskar boarded at Southampton.
  Fact 0.754: Holverson, Mr. Alexander Oskar's destination is New York City.
  Fact 0.755: Holverson, Mr. Alexander Oskar has 38 in the lifeboat or body column.
  Fact 0.756: Holverson, Mrs. Mary Aline (née Towner) was 35 years old.
  Fact 0.757: Holverson, Mrs. Mary Aline (née Towner) is from New York City.
  Fact 0.758: Holverson, Mrs. Mary Aline (née Towner) boarded at Southampton.
  Fact 0.759: Holverson, Mrs. Mary Aline (née Towner)'s destination is New York City.
  Fact 0.760: Holverson, Mrs. Mary Aline (née Towner) has 8 in the lifeboat or body column.
  Fact 0.761: Homer, Mr. Harry (alias E. Haven) was 40 years old.
  Fact 0.762: Homer, Mr. Harry (alias E. Haven) is from Indianapolis, Indiana, US.
  Fact 0.763: Homer, Mr. Harry (alias E. Haven) boarded at Southampton.
  Fact 0.764: Homer, Mr. Harry (alias E. Haven)'s destination is Indianapolis, Indiana, US.
  Fact 0.765: Homer, Mr. Harry (alias E. Haven) has 15 in the lifeboat or body column.
  Fact 0.766: Hoyt, Mr. Frederick Maxfield was 35 years old.
  Fact 0.767: Hoyt, Mr. Frederick Maxfield is from New York City.
  Fact 0.768: Hoyt, Mr. Frederick Maxfield boarded at Southampton.
  Fact 0.769: Hoyt, Mr. Frederick Maxfield's destination is Stamford, Connecticut, US.
  Fact 0.770: Hoyt, Mrs. Jane Anne (née Forby) was 31 years old.
  Fact 0.771: Hoyt, Mrs. Jane Anne (née Forby) is from New York City.
  Fact 0.772: Hoyt, Mrs. Jane Anne (née Forby) boarded at Southampton.
  Fact 0.773: Hoyt, Mrs. Jane Anne (née Forby)'s destination is Stamford, Connecticut, US.
  Fact 0.774: Hoyt, Mr. William Fisher was 42 years old.
  Fact 0.775: Hoyt, Mr. William Fisher is from New York City.
  Fact 0.776: Hoyt, Mr. William Fisher boarded at Cherbourg.
  Fact 0.777: Hoyt, Mr. William Fisher's destination is New York City.
  Fact 0.778: Hoyt, Mr. William Fisher has 14 in the lifeboat or body column.
  Fact 0.779: Isham, Miss Ann Elizabeth was 50 years old.
  Fact 0.780: Isham, Miss Ann Elizabeth is from Chicago, Illinois, US.
  Fact 0.781: Isham, Miss Ann Elizabeth boarded at Cherbourg.
  Fact 0.782: Isham, Miss Ann Elizabeth's destination is Chicago, Illinois, US.
  Fact 0.783: Ismay, Mr. Joseph Bruce was 49 years old.
  Fact 0.784: Ismay, Mr. Joseph Bruce is from Liverpool, Merseyside, England, UK.
  Fact 0.785: Ismay, Mr. Joseph Bruce boarded at Southampton.
  Fact 0.786: Ismay, Mr. Joseph Bruce's destination is New York City.
  Fact 0.787: Mr. John Richard Fry was 39 years old.
  Fact 0.788: Mr. John Richard Fry is listed as the valet.
  Fact 0.789: Mr. John Richard Fry is from Liverpool, Merseyside, England, UK.
  Fact 0.790: Mr. John Richard Fry boarded at Southampton.
  Fact 0.791: Mr. John Richard Fry's destination is New York City.
  Fact 0.792: Mr. William Henry Harrison was 45 years old.
  Fact 0.793: Mr. William Henry Harrison is listed as the secretary.
  Fact 0.794: Mr. William Henry Harrison is from Wallasey, Merseyside, England, UK.
  Fact 0.795: Mr. William Henry Harrison boarded at Southampton.
  Fact 0.796: Mr. William Henry Harrison's destination is New York City.
  Fact 0.797: Mr. William Henry Harrison has 110 in the lifeboat or body column.
  Fact 0.798: Johnson, Mrs. Madeline Eliza (née Austin) was 40 years old.
  Fact 0.799: Johnson, Mrs. Madeline Eliza (née Austin) is from London, England, UK.
  Fact 0.800: Johnson, Mrs. Madeline Eliza (née Austin) boarded at Southampton.
  Fact 0.801: Johnson, Mrs. Madeline Eliza (née Austin)'s destination is New York, New York, US.
  Fact 0.802: Johnson, Mrs. Madeline Eliza (née Austin) has 6 in the lifeboat or body column.
  Fact 0.803: Johnson (later changed to Langston), Mr. Samuel James was 10 years old.
  Fact 0.804: Johnson (later changed to Langston), Mr. Samuel James is from London, England, UK.
  Fact 0.805: Johnson (later changed to Langston), Mr. Samuel James boarded at Southampton.
  Fact 0.806: Johnson (later changed to Langston), Mr. Samuel James's destination is New York, New York, US.
  Fact 0.807: Jones, Mr. Charles Cresson was 46 years old.
  Fact 0.808: Jones, Mr. Charles Cresson is from Bennington, Vermont, US.
  Fact 0.809: Jones, Mr. Charles Cresson boarded at Southampton.
  Fact 0.810: Jones, Mr. Charles Cresson's destination is Bennington, Vermont, US.
  Fact 0.811: Jones, Mr. Charles Cresson has 80 in the lifeboat or body column.
  Fact 0.812: Julian, Mr. Henry Forbes was 50 years old.
  Fact 0.813: Julian, Mr. Henry Forbes is from Torquay, Devon, England, UK.
  Fact 0.814: Julian, Mr. Henry Forbes boarded at Southampton.
  Fact 0.815: Julian, Mr. Henry Forbes's destination is San Francisco, US.
  Fact 0.816: Kent, Mr. Edward Austin was 58 years old.
  Fact 0.817: Kent, Mr. Edward Austin is from Buffalo, New York, US.
  Fact 0.818: Kent, Mr. Edward Austin boarded at Cherbourg.
  Fact 0.819: Kent, Mr. Edward Austin's destination is Buffalo, New York, US.
  Fact 0.820: Kent, Mr. Edward Austin has 258 in the lifeboat or body column.
  Fact 0.821: Kenyon, Mr. Frederick R. was 41 years old.
  Fact 0.822: Kenyon, Mr. Frederick R. is from Pittsburgh, Pennsylvania, US.
  Fact 0.823: Kenyon, Mr. Frederick R. boarded at Southampton.
  Fact 0.824: Kenyon, Mr. Frederick R.'s destination is Pittsburgh, Pennsylvania, US.
  Fact 0.825: Kenyon, Mrs. Marion (née Stauffer) was 31 years old.
  Fact 0.826: Kenyon, Mrs. Marion (née Stauffer) is from Pittsburgh, Pennsylvania, US.
  Fact 0.827: Kenyon, Mrs. Marion (née Stauffer) boarded at Southampton.
  Fact 0.828: Kenyon, Mrs. Marion (née Stauffer)'s destination is Pittsburgh, Pennsylvania, US.
  Fact 0.829: Kenyon, Mrs. Marion (née Stauffer) has 8 in the lifeboat or body column.
  Fact 0.830: Kimball, Mr. Edwin Nelson Jr. was 42 years old.
  Fact 0.831: Kimball, Mr. Edwin Nelson Jr. is from Boston, Massachusetts, US.
  Fact 0.832: Kimball, Mr. Edwin Nelson Jr. boarded at Southampton.
  Fact 0.833: Kimball, Mr. Edwin Nelson Jr.'s destination is Boston, Massachusetts, US.
  Fact 0.834: Kimball, Mr. Edwin Nelson Jr. has 5 in the lifeboat or body column.
  Fact 0.835: Kimball, Mrs. Gertrude (née Parsons) was 45 years old.
  Fact 0.836: Kimball, Mrs. Gertrude (née Parsons) is from Boston, Massachusetts, US.
  Fact 0.837: Kimball, Mrs. Gertrude (née Parsons) boarded at Southampton.
  Fact 0.838: Kimball, Mrs. Gertrude (née Parsons)'s destination is Boston, Massachusetts, US.
  Fact 0.839: Kimball, Mrs. Gertrude (née Parsons) has 5 in the lifeboat or body column.
  Fact 0.840: Klaber, Mr. Herman was 45 years old.
  Fact 0.841: Klaber, Mr. Herman is from Portland, Oregon, US.
  Fact 0.842: Klaber, Mr. Herman boarded at Southampton.
  Fact 0.843: Klaber, Mr. Herman's destination is Portland, Oregon, US.
  Fact 0.844: Langston, Ms. Maxine Jane was 25 years old.
  Fact 0.845: Langston, Ms. Maxine Jane is from Duncansville, Pennsylvania, US.
  Fact 0.846: Langston, Ms. Maxine Jane boarded at Southampton.
  Fact 0.847: Langston, Ms. Maxine Jane's destination is Duncansivlle, Pennsylvania, US.
  Fact 0.848: Lambert-Williams, Mr. Fletcher Fellows was 43 years old.
  Fact 0.849: Lambert-Williams, Mr. Fletcher Fellows is from London, England, UK.
  Fact 0.850: Lambert-Williams, Mr. Fletcher Fellows boarded at Southampton.
  Fact 0.851: Lambert-Williams, Mr. Fletcher Fellows's destination is Newark, New Jersey, US.
  Fact 0.852: Leader, Dr. Alice (née Farnham) was 49 years old.
  Fact 0.853: Leader, Dr. Alice (née Farnham) is from New York City.
  Fact 0.854: Leader, Dr. Alice (née Farnham) boarded at Southampton.
  Fact 0.855: Leader, Dr. Alice (née Farnham)'s destination is New York City.
  Fact 0.856: Leader, Dr. Alice (née Farnham) has 8 in the lifeboat or body column.
  Fact 0.857: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) was 27 years old.
  Fact 0.858: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) is from Vancouver, British Columbia, Canada.
  Fact 0.859: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) boarded at Southampton.
  Fact 0.860: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes)'s destination is Vancouver, British Columbia, Canada.
  Fact 0.861: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) has 8 in the lifeboat or body column.
  Fact 0.862: Miss Roberta Elizabeth Mary "Cissy" Maioni was 20 years old.
  Fact 0.863: Miss Roberta Elizabeth Mary "Cissy" Maioni is listed as the maid.
  Fact 0.864: Miss Roberta Elizabeth Mary "Cissy" Maioni is from Vancouver, British Columbia, Canada.
  Fact 0.865: Miss Roberta Elizabeth Mary "Cissy" Maioni boarded at Southampton.
  Fact 0.866: Miss Roberta Elizabeth Mary "Cissy" Maioni's destination is Vancouver, British Columbia, Canada.
  Fact 0.867: Miss Roberta Elizabeth Mary "Cissy" Maioni has 8 in the lifeboat or body column.
  Fact 0.868: Lewy, Mr. Ervin G. was 31 years old.
  Fact 0.869: Lewy, Mr. Ervin G. is from Chicago, Illinois, US.
  Fact 0.870: Lewy, Mr. Ervin G. boarded at Cherbourg.
  Fact 0.871: Lewy, Mr. Ervin G.'s destination is Chicago, Illinois, US.
  Fact 0.872: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) was 42 years old.
  Fact 0.873: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) is from Jordanstorp, Södermanland, Sweden.
  Fact 0.874: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) boarded at Southampton.
  Fact 0.875: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey)'s destination is New York City.
  Fact 0.876: Lindström, Mrs. Sigrid (née Posse) was 55 years old.
  Fact 0.877: Lindström, Mrs. Sigrid (née Posse) is from Stockholm, Sweden.
  Fact 0.878: Lindström, Mrs. Sigrid (née Posse) boarded at Cherbourg.
  Fact 0.879: Lindström, Mrs. Sigrid (née Posse)'s destination is New York City.
  Fact 0.880: Lindström, Mrs. Sigrid (née Posse) has 6 in the lifeboat or body column.
  Fact 0.881: Lines, Mrs. Elizabeth Lindsey (née James) was 50 years old.
  Fact 0.882: Lines, Mrs. Elizabeth Lindsey (née James) is from Paris, France.
  Fact 0.883: Lines, Mrs. Elizabeth Lindsey (née James) boarded at Cherbourg.
  Fact 0.884: Lines, Mrs. Elizabeth Lindsey (née James)'s destination is Hanover, New Hampshire, US.
  Fact 0.885: Lines, Mrs. Elizabeth Lindsey (née James) has 9 in the lifeboat or body column.
  Fact 0.886: Lines, Miss Mary Conover was 16 years old.
  Fact 0.887: Lines, Miss Mary Conover is from Paris, France.
  Fact 0.888: Lines, Miss Mary Conover boarded at Cherbourg.
  Fact 0.889: Lines, Miss Mary Conover's destination is Hanover, New Hampshire, US.
  Fact 0.890: Lines, Miss Mary Conover has 9 in the lifeboat or body column.
  Fact 0.891: Long, Mr. Milton Clyde was 29 years old.
  Fact 0.892: Long, Mr. Milton Clyde is from Springfield, Massachusetts, US.
  Fact 0.893: Long, Mr. Milton Clyde boarded at Southampton.
  Fact 0.894: Long, Mr. Milton Clyde's destination is Springfield, Massachusetts, US.
  Fact 0.895: Long, Mr. Milton Clyde has 126 in the lifeboat or body column.
  Fact 0.896: Longley, Miss Gretchen Fiske was 21 years old.
  Fact 0.897: Longley, Miss Gretchen Fiske is from Hudson, New York, US.
  Fact 0.898: Longley, Miss Gretchen Fiske boarded at Cherbourg.
  Fact 0.899: Longley, Miss Gretchen Fiske's destination is Hudson, New York, US.
  Fact 0.900: Longley, Miss Gretchen Fiske has 10 in the lifeboat or body column.
  Fact 0.901: Loring, Mr. Joseph Holland was 30 years old.
  Fact 0.902: Loring, Mr. Joseph Holland is from London, England, UK.
  Fact 0.903: Loring, Mr. Joseph Holland boarded at Southampton.
  Fact 0.904: Loring, Mr. Joseph Holland's destination is New York City.
  Fact 0.905: Madill, Miss Georgette Alexandra was 16 years old.
  Fact 0.906: Madill, Miss Georgette Alexandra is from St. Louis, Missouri, US.
  Fact 0.907: Madill, Miss Georgette Alexandra boarded at Southampton.
  Fact 0.908: Madill, Miss Georgette Alexandra's destination is St. Louis, Missouri, US.
  Fact 0.909: Maguire, Mr. John Edward was 30 years old.
  Fact 0.910: Maguire, Mr. John Edward is from Brockton, Massachusetts, US.
  Fact 0.911: Maguire, Mr. John Edward boarded at Southampton.
  Fact 0.912: Maguire, Mr. John Edward's destination is Brockton, Massachusetts, US.
  Fact 0.913: Maréchal, Mr. Pierre, Sr. was 28 years old.
  Fact 0.914: Maréchal, Mr. Pierre, Sr. is from Paris, France.
  Fact 0.915: Maréchal, Mr. Pierre, Sr. boarded at Cherbourg.
  Fact 0.916: Maréchal, Mr. Pierre, Sr.'s destination is New York City.
  Fact 0.917: Maréchal, Mr. Pierre, Sr. has 7 in the lifeboat or body column.
  Fact 0.918: Marvin, Mr. Daniel Warner was 18 years old.
  Fact 0.919: Marvin, Mr. Daniel Warner is from New York City.
  Fact 0.920: Marvin, Mr. Daniel Warner boarded at Southampton.
  Fact 0.921: Marvin, Mr. Daniel Warner's destination is New York City.
  Fact 0.922: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) was 18 years old.
  Fact 0.923: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) is from New York City.
  Fact 0.924: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) boarded at Southampton.
  Fact 0.925: Marvin, Mrs. Mary Graham Carmichael (née Farquarson)'s destination is New York City.
  Fact 0.926: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) has 10 in the lifeboat or body column.
  Fact 0.927: Mayné, Miss Bertha Antonine was 24 years old.
  Fact 0.928: Mayné, Miss Bertha Antonine is from Brussels, Belgium.
  Fact 0.929: Mayné, Miss Bertha Antonine boarded at Cherbourg.
  Fact 0.930: Mayné, Miss Bertha Antonine's destination is Montreal, Quebec, Canada.
  Fact 0.931: Mayné, Miss Bertha Antonine has 6 in the lifeboat or body column.
  Fact 0.932: McCaffry, Mr. Thomas Francis was 46 years old.
  Fact 0.933: McCaffry, Mr. Thomas Francis is from Vancouver, British Columbia, Canada.
  Fact 0.934: McCaffry, Mr. Thomas Francis boarded at Southampton.
  Fact 0.935: McCaffry, Mr. Thomas Francis's destination is Vancouver, British Columbia, Canada.
  Fact 0.936: McCaffry, Mr. Thomas Francis has 292 in the lifeboat or body column.
  Fact 0.937: McCarthy, Mr. Timothy J. was 54 years old.
  Fact 0.938: McCarthy, Mr. Timothy J. is from Dorchester, Massachusetts, US.
  Fact 0.939: McCarthy, Mr. Timothy J. boarded at Southampton.
  Fact 0.940: McCarthy, Mr. Timothy J.'s destination is Dorchester, Massachusetts, US.
  Fact 0.941: McCarthy, Mr. Timothy J. has 175 in the lifeboat or body column.
  Fact 0.942: McGough, Mr. James Robert was 35 years old.
  Fact 0.943: McGough, Mr. James Robert is from Philadelphia, Pennsylvania, US.
  Fact 0.944: McGough, Mr. James Robert boarded at Southampton.
  Fact 0.945: McGough, Mr. James Robert's destination is Philadelphia, Pennsylvania, US.
  Fact 0.946: McGough, Mr. James Robert has 7 in the lifeboat or body column.
  Fact 0.947: Meyer, Mr. Edgar Joseph was 28 years old.
  Fact 0.948: Meyer, Mr. Edgar Joseph is from New York City.
  Fact 0.949: Meyer, Mr. Edgar Joseph boarded at Cherbourg.
  Fact 0.950: Meyer, Mr. Edgar Joseph's destination is New York City.
  Fact 0.951: Meyer, Mrs. Leila (née Saks) was 25 years old.
  Fact 0.952: Meyer, Mrs. Leila (née Saks) is from New York City.
  Fact 0.953: Meyer, Mrs. Leila (née Saks) boarded at Cherbourg.
  Fact 0.954: Meyer, Mrs. Leila (née Saks)'s destination is New York City.
  Fact 0.955: Meyer, Mrs. Leila (née Saks) has 6 in the lifeboat or body column.
  Fact 0.956: Millet, Mr. Francis Davis was 65 years old.
  Fact 0.957: Millet, Mr. Francis Davis is from East Bridgewater, Massachusetts, US.
  Fact 0.958: Millet, Mr. Francis Davis boarded at Cherbourg.
  Fact 0.959: Millet, Mr. Francis Davis's destination is East Bridgewater, Massachusetts, US.
  Fact 0.960: Millet, Mr. Francis Davis has 249 in the lifeboat or body column.
  Fact 0.961: Minahan, Dr. William Edward was 44 years old.
  Fact 0.962: Minahan, Dr. William Edward is from Fond du Lac, Wisconsin, US.
  Fact 0.963: Minahan, Dr. William Edward boarded at Southampton.
  Fact 0.964: Minahan, Dr. William Edward's destination is Fond du Lac, Wisconsin, US.
  Fact 0.965: Minahan, Dr. William Edward has 230 in the lifeboat or body column.
  Fact 0.966: Minahan, Mrs. Lillian E. (née Thorpe) was 37 years old.
  Fact 0.967: Minahan, Mrs. Lillian E. (née Thorpe) is from Fond du Lac, Wisconsin, US.
  Fact 0.968: Minahan, Mrs. Lillian E. (née Thorpe) boarded at Southampton.
  Fact 0.969: Minahan, Mrs. Lillian E. (née Thorpe)'s destination is Fond du Lac, Wisconsin, US.
  Fact 0.970: Minahan, Mrs. Lillian E. (née Thorpe) has 14 in the lifeboat or body column.
  Fact 0.971: Minahan, Miss Daisy E. was 33 years old.
  Fact 0.972: Minahan, Miss Daisy E. is from Fond du Lac, Wisconsin, US.
  Fact 0.973: Minahan, Miss Daisy E. boarded at Southampton.
  Fact 0.974: Minahan, Miss Daisy E.'s destination is Fond du Lac, Wisconsin, US.
  Fact 0.975: Minahan, Miss Daisy E. has 14 in the lifeboat or body column.
  Fact 0.976: Mock, Mr. Philipp Edmund was 30 years old.
  Fact 0.977: Mock, Mr. Philipp Edmund is from New York City.
  Fact 0.978: Mock, Mr. Philipp Edmund boarded at Cherbourg.
  Fact 0.979: Mock, Mr. Philipp Edmund's destination is New York City.
  Fact 0.980: Mock, Mr. Philipp Edmund has 11 in the lifeboat or body column.
  Fact 0.981: Molson, Mr. Harry Markland was 55 years old.
  Fact 0.982: Molson, Mr. Harry Markland is from Montreal, Quebec, Canada.
  Fact 0.983: Molson, Mr. Harry Markland boarded at Southampton.
  Fact 0.984: Molson, Mr. Harry Markland's destination is Montreal, Quebec, Canada.
  Fact 0.985: Moore, Mr. Clarence Bloomfield was 47 years old.
  Fact 0.986: Moore, Mr. Clarence Bloomfield is from Washington, D.C., US.
  Fact 0.987: Moore, Mr. Clarence Bloomfield boarded at Southampton.
  Fact 0.988: Moore, Mr. Clarence Bloomfield's destination is Washington, D.C., US.
  Fact 0.989: Mr. Charles Henry Harrington was 37 years old.
  Fact 0.990: Mr. Charles Henry Harrington is listed as the valet.
  Fact 0.991: Mr. Charles Henry Harrington is from Washington, D.C., US.
  Fact 0.992: Mr. Charles Henry Harrington boarded at Southampton.
  Fact 0.993: Mr. Charles Henry Harrington's destination is Washington, D.C., US.
  Fact 0.994: Natsch, Mr. Charles was 36 years old.
  Fact 0.995: Natsch, Mr. Charles is from Brooklyn, New York, US.
  Fact 0.996: Natsch, Mr. Charles boarded at Cherbourg.
  Fact 0.997: Natsch, Mr. Charles's destination is Brooklyn, New York, US.
  Fact 0.998: Newell, Mr. Arthur Webster was 58 years old.
  Fact 0.999: Newell, Mr. Arthur Webster is from Lexington, Massachusetts, US.
  Fact 0.1000: Newell, Mr. Arthur Webster boarded at Cherbourg.
  Fact 0.1001: Newell, Mr. Arthur Webster's destination is Lexington, Massachusetts, US.
  Fact 0.1002: Newell, Mr. Arthur Webster has 122 in the lifeboat or body column.
  Fact 0.1003: Newell, Miss Madeleine was 31 years old.
  Fact 0.1004: Newell, Miss Madeleine is from Lexington, Massachusetts, US.
  Fact 0.1005: Newell, Miss Madeleine boarded at Cherbourg.
  Fact 0.1006: Newell, Miss Madeleine's destination is Lexington, Massachusetts, US.
  Fact 0.1007: Newell, Miss Madeleine has 6 in the lifeboat or body column.
  Fact 0.1008: Newell, Miss Marjorie Anne was 23 years old.
  Fact 0.1009: Newell, Miss Marjorie Anne is from Lexington, Massachusetts, US.
  Fact 0.1010: Newell, Miss Marjorie Anne boarded at Cherbourg.
  Fact 0.1011: Newell, Miss Marjorie Anne's destination is Lexington, Massachusetts, US.
  Fact 0.1012: Newell, Miss Marjorie Anne has 6 in the lifeboat or body column.
  Fact 0.1013: Newsom, Miss Helen Monypeny was 19 years old.
  Fact 0.1014: Newsom, Miss Helen Monypeny is from New York City.
  Fact 0.1015: Newsom, Miss Helen Monypeny boarded at Southampton.
  Fact 0.1016: Newsom, Miss Helen Monypeny's destination is New York City.
  Fact 0.1017: Newsom, Miss Helen Monypeny has 5 in the lifeboat or body column.
  Fact 0.1018: Nicholson, Mr. Arthur Ernest was 64 years old.
  Fact 0.1019: Nicholson, Mr. Arthur Ernest is from Shanklin, Isle of Wight, England, UK.
  Fact 0.1020: Nicholson, Mr. Arthur Ernest boarded at Southampton.
  Fact 0.1021: Nicholson, Mr. Arthur Ernest's destination is New York City.
  Fact 0.1022: Nicholson, Mr. Arthur Ernest has 263 in the lifeboat or body column.
  Fact 0.1023: Nourney, Mr. Alfred was 20 years old.
  Fact 0.1024: Nourney, Mr. Alfred is from Cologne, German Empire.
  Fact 0.1025: Nourney, Mr. Alfred boarded at Cherbourg.
  Fact 0.1026: Nourney, Mr. Alfred's destination is New York City.
  Fact 0.1027: Nourney, Mr. Alfred has 7 in the lifeboat or body column.
  Fact 0.1028: Omont, Mr. Alfred Fernand was 29 years old.
  Fact 0.1029: Omont, Mr. Alfred Fernand is from Le Havre, France.
  Fact 0.1030: Omont, Mr. Alfred Fernand boarded at Cherbourg.
  Fact 0.1031: Omont, Mr. Alfred Fernand's destination is New York City.
  Fact 0.1032: Omont, Mr. Alfred Fernand has 7 in the lifeboat or body column.
  Fact 0.1033: Ostby, Mr. Engelhart Cornelius was 64 years old.
  Fact 0.1034: Ostby, Mr. Engelhart Cornelius is from Providence, Rhode Island, US.
  Fact 0.1035: Ostby, Mr. Engelhart Cornelius boarded at Southampton.
  Fact 0.1036: Ostby, Mr. Engelhart Cornelius's destination is Providence, Rhode Island, US.
  Fact 0.1037: Ostby, Mr. Engelhart Cornelius has 234 in the lifeboat or body column.
  Fact 0.1038: Ostby, Miss Helene Ragnhild was 22 years old.
  Fact 0.1039: Ostby, Miss Helene Ragnhild is from Providence, Rhode Island, US.
  Fact 0.1040: Ostby, Miss Helene Ragnhild boarded at Southampton.
  Fact 0.1041: Ostby, Miss Helene Ragnhild's destination is Providence, Rhode Island, US.
  Fact 0.1042: Ostby, Miss Helene Ragnhild has 5 in the lifeboat or body column.
  Fact 0.1043: Ovies y Rodriguez, Mr. Servando José Florentino was 36 years old.
  Fact 0.1044: Ovies y Rodriguez, Mr. Servando José Florentino is from Havana, Cuba.
  Fact 0.1045: Ovies y Rodriguez, Mr. Servando José Florentino boarded at Cherbourg.
  Fact 0.1046: Ovies y Rodriguez, Mr. Servando José Florentino's destination is Havana, Cuba.
  Fact 0.1047: Ovies y Rodriguez, Mr. Servando José Florentino has 189 in the lifeboat or body column.
  Fact 0.1048: Parr, Mr. William Henry Marsh was 29 years old.
  Fact 0.1049: Parr, Mr. William Henry Marsh is from Belfast, Ireland, UK.
  Fact 0.1050: Parr, Mr. William Henry Marsh boarded at Belfast.
  Fact 0.1051: Parr, Mr. William Henry Marsh's destination is New York City.
  Fact 0.1052: Partner, Mr. Austin was 40 years old.
  Fact 0.1053: Partner, Mr. Austin is from Tolworth, London, England, UK.
  Fact 0.1054: Partner, Mr. Austin boarded at Southampton.
  Fact 0.1055: Partner, Mr. Austin's destination is Toronto, Ontario, Canada.
  Fact 0.1056: Partner, Mr. Austin has 166 in the lifeboat or body column.
  Fact 0.1057: Pears, Mr. Thomas Clinton was 29 years old.
  Fact 0.1058: Pears, Mr. Thomas Clinton is from Isleworth, London, England, UK.
  Fact 0.1059: Pears, Mr. Thomas Clinton boarded at Southampton.
  Fact 0.1060: Pears, Mr. Thomas Clinton's destination is New York City.
  Fact 0.1061: Pears, Mrs. Edith (née Wearne) was 22 years old.
  Fact 0.1062: Pears, Mrs. Edith (née Wearne) is from Isleworth, London, England, UK.
  Fact 0.1063: Pears, Mrs. Edith (née Wearne) boarded at Southampton.
  Fact 0.1064: Pears, Mrs. Edith (née Wearne)'s destination is New York City.
  Fact 0.1065: Pears, Mrs. Edith (née Wearne) has 8 in the lifeboat or body column.
  Fact 0.1066: Peñasco y Castellana, Mr. Victor was 24 years old.
  Fact 0.1067: Peñasco y Castellana, Mr. Victor is from Madrid, Spain.
  Fact 0.1068: Peñasco y Castellana, Mr. Victor boarded at Cherbourg.
  Fact 0.1069: Peñasco y Castellana, Mr. Victor's destination is New York City.
  Fact 0.1070: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) was 22 years old.
  Fact 0.1071: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) is from Madrid, Spain.
  Fact 0.1072: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) boarded at Cherbourg.
  Fact 0.1073: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo)'s destination is New York City.
  Fact 0.1074: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) has 8 in the lifeboat or body column.
  Fact 0.1075: Doña Fermina Oliva y Ocana was 39 years old.
  Fact 0.1076: Doña Fermina Oliva y Ocana is listed as the maid.
  Fact 0.1077: Doña Fermina Oliva y Ocana is from Madrid, Spain.
  Fact 0.1078: Doña Fermina Oliva y Ocana boarded at Cherbourg.
  Fact 0.1079: Doña Fermina Oliva y Ocana's destination is New York City.
  Fact 0.1080: Doña Fermina Oliva y Ocana has 8 in the lifeboat or body column.
  Fact 0.1081: Peuchen, Major Arthur Godfrey was 52 years old.
  Fact 0.1082: Peuchen, Major Arthur Godfrey is from Toronto, Ontario, Canada.
  Fact 0.1083: Peuchen, Major Arthur Godfrey boarded at Southampton.
  Fact 0.1084: Peuchen, Major Arthur Godfrey's destination is Toronto, Ontario, Canada.
  Fact 0.1085: Peuchen, Major Arthur Godfrey has 6 in the lifeboat or body column.
  Fact 0.1086: Porter, Mr. Walter Chamberlain was 46 years old.
  Fact 0.1087: Porter, Mr. Walter Chamberlain is from Worcester, Massachusetts, US.
  Fact 0.1088: Porter, Mr. Walter Chamberlain boarded at Southampton.
  Fact 0.1089: Porter, Mr. Walter Chamberlain's destination is Worcester, Massachusetts, US.
  Fact 0.1090: Porter, Mr. Walter Chamberlain has 207 in the lifeboat or body column.
  Fact 0.1091: Potter, Mrs. Lily Alexenia (née Wilson) was 56 years old.
  Fact 0.1092: Potter, Mrs. Lily Alexenia (née Wilson) is from Philadelphia, Pennsylvania, US.
  Fact 0.1093: Potter, Mrs. Lily Alexenia (née Wilson) boarded at Cherbourg.
  Fact 0.1094: Potter, Mrs. Lily Alexenia (née Wilson)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.1095: Potter, Mrs. Lily Alexenia (née Wilson) has 7 in the lifeboat or body column.
  Fact 0.1096: Reuchlin, The Honourable Mr. Johan George was 38 years old.
  Fact 0.1097: Reuchlin, The Honourable Mr. Johan George is from Rotterdam, The Netherlands.
  Fact 0.1098: Reuchlin, The Honourable Mr. Johan George boarded at Southampton.
  Fact 0.1099: Reuchlin, The Honourable Mr. Johan George's destination is New York City.
  Fact 0.1100: Rheims, Mr. George Alexander Lucien was 36 years old.
  Fact 0.1101: Rheims, Mr. George Alexander Lucien is from Paris, France.
  Fact 0.1102: Rheims, Mr. George Alexander Lucien boarded at Southampton.
  Fact 0.1103: Rheims, Mr. George Alexander Lucien's destination is New York City.
  Fact 0.1104: Robert, Mrs. Elisabeth Walton (née McMillan) was 43 years old.
  Fact 0.1105: Robert, Mrs. Elisabeth Walton (née McMillan) is from St. Louis, Missouri, US.
  Fact 0.1106: Robert, Mrs. Elisabeth Walton (née McMillan) boarded at Southampton.
  Fact 0.1107: Robert, Mrs. Elisabeth Walton (née McMillan)'s destination is St. Louis, Missouri, US.
  Fact 0.1108: Miss Emilie Kreuchen was 29 years old.
  Fact 0.1109: Miss Emilie Kreuchen is listed as the maid.
  Fact 0.1110: Miss Emilie Kreuchen is from St. Louis, Missouri, US.
  Fact 0.1111: Miss Emilie Kreuchen boarded at Southampton.
  Fact 0.1112: Miss Emilie Kreuchen's destination is St. Louis, Missouri, US.
  Fact 0.1113: Roebling, Mr. Washington Augustus II was 31 years old.
  Fact 0.1114: Roebling, Mr. Washington Augustus II is from Trenton, New Jersey, US.
  Fact 0.1115: Roebling, Mr. Washington Augustus II boarded at Southampton.
  Fact 0.1116: Roebling, Mr. Washington Augustus II's destination is Trenton, New Jersey, US.
  Fact 0.1117: Romaine, Mr. Charles Hallance (alias C. Rolmane) was 45 years old.
  Fact 0.1118: Romaine, Mr. Charles Hallance (alias C. Rolmane) is from New York City.
  Fact 0.1119: Romaine, Mr. Charles Hallance (alias C. Rolmane) boarded at Southampton.
  Fact 0.1120: Romaine, Mr. Charles Hallance (alias C. Rolmane)'s destination is New York City.
  Fact 0.1121: Romaine, Mr. Charles Hallance (alias C. Rolmane) has 9 in the lifeboat or body column.
  Fact 0.1122: Rood, Mr. Hugh Roscoe was 38 years old.
  Fact 0.1123: Rood, Mr. Hugh Roscoe is from Seattle, Washington, US.
  Fact 0.1124: Rood, Mr. Hugh Roscoe boarded at Southampton.
  Fact 0.1125: Rood, Mr. Hugh Roscoe's destination is Seattle, Washington, US.
  Fact 0.1126: Rosenbaum, Miss Edith Louise was 34 years old.
  Fact 0.1127: Rosenbaum, Miss Edith Louise is from Paris, France.
  Fact 0.1128: Rosenbaum, Miss Edith Louise boarded at Cherbourg.
  Fact 0.1129: Rosenbaum, Miss Edith Louise's destination is Cincinnati, Ohio, US.
  Fact 0.1130: Rosenbaum, Miss Edith Louise has 11 in the lifeboat or body column.
  Fact 0.1131: Rosenshine, Mr. George (alias George Thorne) was 46 years old.
  Fact 0.1132: Rosenshine, Mr. George (alias George Thorne) is from New York City.
  Fact 0.1133: Rosenshine, Mr. George (alias George Thorne) boarded at Cherbourg.
  Fact 0.1134: Rosenshine, Mr. George (alias George Thorne)'s destination is New York City.
  Fact 0.1135: Rosenshine, Mr. George (alias George Thorne) has 16 in the lifeboat or body column.
  Fact 0.1136: Ross, Mr. John Hugo was 36 years old.
  Fact 0.1137: Ross, Mr. John Hugo is from Winnipeg, Manitoba, Canada.
  Fact 0.1138: Ross, Mr. John Hugo boarded at Southampton.
  Fact 0.1139: Ross, Mr. John Hugo's destination is Winnipeg, Manitoba, Canada.
  Fact 0.1140: Rothschild, Mr. Martin was 46 years old.
  Fact 0.1141: Rothschild, Mr. Martin is from New York City.
  Fact 0.1142: Rothschild, Mr. Martin boarded at Cherbourg.
  Fact 0.1143: Rothschild, Mr. Martin's destination is New York City.
  Fact 0.1144: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) was 54 years old.
  Fact 0.1145: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) is from New York City.
  Fact 0.1146: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) boarded at Cherbourg.
  Fact 0.1147: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett)'s destination is New York City.
  Fact 0.1148: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) has 6 in the lifeboat or body column.
  Fact 0.1149: Rowe, Mr. Alfred G. was 59 years old.
  Fact 0.1150: Rowe, Mr. Alfred G. is from Liverpool, Merseyside, England, UK.
  Fact 0.1151: Rowe, Mr. Alfred G. boarded at Southampton.
  Fact 0.1152: Rowe, Mr. Alfred G.'s destination is New York City.
  Fact 0.1153: Rowe, Mr. Alfred G. has 109 in the lifeboat or body column.
  Fact 0.1154: Ryerson, Mr. Arthur Larned was 61 years old.
  Fact 0.1155: Ryerson, Mr. Arthur Larned is from Cooperstown, New York, US.
  Fact 0.1156: Ryerson, Mr. Arthur Larned boarded at Cherbourg.
  Fact 0.1157: Ryerson, Mr. Arthur Larned's destination is Cooperstown, New York, US.
  Fact 0.1158: Ryerson, Mrs. Emily Maria (née Borie) was 48 years old.
  Fact 0.1159: Ryerson, Mrs. Emily Maria (née Borie) is from Cooperstown, New York, US.
  Fact 0.1160: Ryerson, Mrs. Emily Maria (née Borie) boarded at Cherbourg.
  Fact 0.1161: Ryerson, Mrs. Emily Maria (née Borie)'s destination is Cooperstown, New York, US.
  Fact 0.1162: Miss Victorine Chaudanson was 36 years old.
  Fact 0.1163: Miss Victorine Chaudanson is listed as the maid.
  Fact 0.1164: Miss Victorine Chaudanson is from Cooperstown, New York, US.
  Fact 0.1165: Miss Victorine Chaudanson boarded at Cherbourg.
  Fact 0.1166: Miss Victorine Chaudanson's destination is Cooperstown, New York, US.
  Fact 0.1167: Ryerson, Miss Susan Parker "Suzette" was 21 years old.
  Fact 0.1168: Ryerson, Miss Susan Parker "Suzette" is from Cooperstown, New York, US.
  Fact 0.1169: Ryerson, Miss Susan Parker "Suzette" boarded at Cherbourg.
  Fact 0.1170: Ryerson, Miss Susan Parker "Suzette"'s destination is Cooperstown, New York, US.
  Fact 0.1171: Ryerson, Miss Emily Borie was 18 years old.
  Fact 0.1172: Ryerson, Miss Emily Borie is from Cooperstown, New York, US.
  Fact 0.1173: Ryerson, Miss Emily Borie boarded at Cherbourg.
  Fact 0.1174: Ryerson, Miss Emily Borie's destination is Cooperstown, New York, US.
  Fact 0.1175: Ryerson, Master John Borie "Jack" was 13 years old.
  Fact 0.1176: Ryerson, Master John Borie "Jack" is from Cooperstown, New York, US.
  Fact 0.1177: Ryerson, Master John Borie "Jack" boarded at Cherbourg.
  Fact 0.1178: Ryerson, Master John Borie "Jack"'s destination is Cooperstown, New York, US.
  Fact 0.1179: Miss Grace Scott Bowen was 45 years old.
  Fact 0.1180: Miss Grace Scott Bowen is listed as the governess.
  Fact 0.1181: Miss Grace Scott Bowen is from Cooperstown, New York, US.
  Fact 0.1182: Miss Grace Scott Bowen boarded at Cherbourg.
  Fact 0.1183: Miss Grace Scott Bowen's destination is Cooperstown, New York, US.
  Fact 0.1184: Saalfeld, Mr. Adolphe was 47 years old.
  Fact 0.1185: Saalfeld, Mr. Adolphe is from Manchester, England, UK.
  Fact 0.1186: Saalfeld, Mr. Adolphe boarded at Southampton.
  Fact 0.1187: Saalfeld, Mr. Adolphe's destination is New York City.
  Fact 0.1188: Salomon, Mr. Abraham Lincoln was 43 years old.
  Fact 0.1189: Salomon, Mr. Abraham Lincoln is from New York City.
  Fact 0.1190: Salomon, Mr. Abraham Lincoln boarded at Cherbourg.
  Fact 0.1191: Salomon, Mr. Abraham Lincoln's destination is New York City.
  Fact 0.1192: Schabert, Mrs. Emma (née Mock) was 35 years old.
  Fact 0.1193: Schabert, Mrs. Emma (née Mock) is from Hamburg, German Empire.
  Fact 0.1194: Schabert, Mrs. Emma (née Mock) boarded at Southampton.
  Fact 0.1195: Schabert, Mrs. Emma (née Mock)'s destination is New York City.
  Fact 0.1196: Schabert, Mrs. Emma (née Mock) has 11 in the lifeboat or body column.
  Fact 0.1197: Seward, Mr. Frederic Kimber was 34 years old.
  Fact 0.1198: Seward, Mr. Frederic Kimber is from New York City.
  Fact 0.1199: Seward, Mr. Frederic Kimber boarded at Southampton.
  Fact 0.1200: Seward, Mr. Frederic Kimber's destination is New York City.
  Fact 0.1201: Seward, Mr. Frederic Kimber has 7 in the lifeboat or body column.
  Fact 0.1202: Silverthorne, Mr. Spencer Victor was 35 years old.
  Fact 0.1203: Silverthorne, Mr. Spencer Victor is from St. Louis, Missouri, US.
  Fact 0.1204: Silverthorne, Mr. Spencer Victor boarded at Southampton.
  Fact 0.1205: Silverthorne, Mr. Spencer Victor's destination is St. Louis, Missouri, US.
  Fact 0.1206: Silverthorne, Mr. Spencer Victor has 5 in the lifeboat or body column.
  Fact 0.1207: Silvey, Mr. William Baird was 50 years old.
  Fact 0.1208: Silvey, Mr. William Baird is from Duluth, Minnesota, US.
  Fact 0.1209: Silvey, Mr. William Baird boarded at Cherbourg.
  Fact 0.1210: Silvey, Mr. William Baird's destination is Duluth, Minnesota, US.
  Fact 0.1211: Silvey, Mrs. Alice (née Munger) was 39 years old.
  Fact 0.1212: Silvey, Mrs. Alice (née Munger) is from Duluth, Minnesota, US.
  Fact 0.1213: Silvey, Mrs. Alice (née Munger) boarded at Cherbourg.
  Fact 0.1214: Silvey, Mrs. Alice (née Munger)'s destination is Duluth, Minnesota, US.
  Fact 0.1215: Silvey, Mrs. Alice (née Munger) has 11 in the lifeboat or body column.
  Fact 0.1216: Simonius-Blumer, Colonel Alfons was 56 years old.
  Fact 0.1217: Simonius-Blumer, Colonel Alfons is from Basel, Switzerland.
  Fact 0.1218: Simonius-Blumer, Colonel Alfons boarded at Southampton.
  Fact 0.1219: Simonius-Blumer, Colonel Alfons's destination is New York City.
  Fact 0.1220: Sloper, Mr. William Thompson was 28 years old.
  Fact 0.1221: Sloper, Mr. William Thompson is from New Britain, Connecticut, US.
  Fact 0.1222: Sloper, Mr. William Thompson boarded at Southampton.
  Fact 0.1223: Sloper, Mr. William Thompson's destination is New Britain, Connecticut, US.
  Fact 0.1224: Sloper, Mr. William Thompson has 7 in the lifeboat or body column.
  Fact 0.1225: Smart, Mr. John Montgomery was 56 years old.
  Fact 0.1226: Smart, Mr. John Montgomery is from Kildale, North Yorkshire, England, UK.
  Fact 0.1227: Smart, Mr. John Montgomery boarded at Southampton.
  Fact 0.1228: Smart, Mr. John Montgomery's destination is New York City.
  Fact 0.1229: Smith, Mr. James Clinch was 56 years old.
  Fact 0.1230: Smith, Mr. James Clinch is from Paris, France.
  Fact 0.1231: Smith, Mr. James Clinch boarded at Cherbourg.
  Fact 0.1232: Smith, Mr. James Clinch's destination is Long Island, New York, US.
  Fact 0.1233: Smith, Mr. Lucien Philip was 24 years old.
  Fact 0.1234: Smith, Mr. Lucien Philip is from Huntington, West Virginia, US.
  Fact 0.1235: Smith, Mr. Lucien Philip boarded at Cherbourg.
  Fact 0.1236: Smith, Mr. Lucien Philip's destination is Huntington, West Virginia, US.
  Fact 0.1237: Smith, Mrs. Mary Eloise (née Hughes) was 18 years old.
  Fact 0.1238: Smith, Mrs. Mary Eloise (née Hughes) is from Huntington, West Virginia, US.
  Fact 0.1239: Smith, Mrs. Mary Eloise (née Hughes) boarded at Cherbourg.
  Fact 0.1240: Smith, Mrs. Mary Eloise (née Hughes)'s destination is Huntington, West Virginia, US.
  Fact 0.1241: Smith, Mrs. Mary Eloise (née Hughes) has 6 in the lifeboat or body column.
  Fact 0.1242: Smith, Mr. Richard William was 57 years old.
  Fact 0.1243: Smith, Mr. Richard William is from Streatham, London, England, UK.
  Fact 0.1244: Smith, Mr. Richard William boarded at Southampton.
  Fact 0.1245: Smith, Mr. Richard William's destination is New York City.
  Fact 0.1246: Snyder, Mr. John Pillsbury was 24 years old.
  Fact 0.1247: Snyder, Mr. John Pillsbury is from Minneapolis, Minnesota, US.
  Fact 0.1248: Snyder, Mr. John Pillsbury boarded at Southampton.
  Fact 0.1249: Snyder, Mr. John Pillsbury's destination is Minneapolis, Minnesota, US.
  Fact 0.1250: Snyder, Mr. John Pillsbury has 7 in the lifeboat or body column.
  Fact 0.1251: Snyder, Mrs. Nellie (née Stevenson) was 23 years old.
  Fact 0.1252: Snyder, Mrs. Nellie (née Stevenson) is from Minneapolis, Minnesota, US.
  Fact 0.1253: Snyder, Mrs. Nellie (née Stevenson) boarded at Southampton.
  Fact 0.1254: Snyder, Mrs. Nellie (née Stevenson)'s destination is Minneapolis, Minnesota, US.
  Fact 0.1255: Snyder, Mrs. Nellie (née Stevenson) has 7 in the lifeboat or body column.
  Fact 0.1256: Spedden, Mr. Frederic Oakley was 45 years old.
  Fact 0.1257: Spedden, Mr. Frederic Oakley is from Tuxedo Park, New York, US.
  Fact 0.1258: Spedden, Mr. Frederic Oakley boarded at Cherbourg.
  Fact 0.1259: Spedden, Mr. Frederic Oakley's destination is Tuxedo Park, New York, US.
  Fact 0.1260: Spedden, Mrs. Margaretta Corning (née Stone) was 39 years old.
  Fact 0.1261: Spedden, Mrs. Margaretta Corning (née Stone) is from Tuxedo Park, New York, US.
  Fact 0.1262: Spedden, Mrs. Margaretta Corning (née Stone) boarded at Cherbourg.
  Fact 0.1263: Spedden, Mrs. Margaretta Corning (née Stone)'s destination is Tuxedo Park, New York, US.
  Fact 0.1264: Miss Helen Alice Wilson was 31 years old.
  Fact 0.1265: Miss Helen Alice Wilson is listed as the maid.
  Fact 0.1266: Miss Helen Alice Wilson is from Tuxedo Park, New York, US.
  Fact 0.1267: Miss Helen Alice Wilson boarded at Cherbourg.
  Fact 0.1268: Miss Helen Alice Wilson's destination is Tuxedo Park, New York, US.
  Fact 0.1269: Spedden, Master Robert Douglas was 6 years old.
  Fact 0.1270: Spedden, Master Robert Douglas is from Tuxedo Park, New York, US.
  Fact 0.1271: Spedden, Master Robert Douglas boarded at Cherbourg.
  Fact 0.1272: Spedden, Master Robert Douglas's destination is Tuxedo Park, New York, US.
  Fact 0.1273: Miss Elizabeth Margaret Burns was 41 years old.
  Fact 0.1274: Miss Elizabeth Margaret Burns is listed as the nurse.
  Fact 0.1275: Miss Elizabeth Margaret Burns is from Tuxedo Park, New York, US.
  Fact 0.1276: Miss Elizabeth Margaret Burns boarded at Cherbourg.
  Fact 0.1277: Miss Elizabeth Margaret Burns's destination is Tuxedo Park, New York, US.
  Fact 0.1278: Spencer, Mr. William Augustus was 57 years old.
  Fact 0.1279: Spencer, Mr. William Augustus is from New York City.
  Fact 0.1280: Spencer, Mr. William Augustus boarded at Cherbourg.
  Fact 0.1281: Spencer, Mr. William Augustus's destination is New York City.
  Fact 0.1282: Spencer, Mrs. Marie Eugénie (née Demougeot) was 45 years old.
  Fact 0.1283: Spencer, Mrs. Marie Eugénie (née Demougeot) is from New York City.
  Fact 0.1284: Spencer, Mrs. Marie Eugénie (née Demougeot) boarded at Cherbourg.
  Fact 0.1285: Spencer, Mrs. Marie Eugénie (née Demougeot)'s destination is New York City.
  Fact 0.1286: Spencer, Mrs. Marie Eugénie (née Demougeot) has 6 in the lifeboat or body column.
  Fact 0.1287: Miss Eugenie Elise Lurette was 59 years old.
  Fact 0.1288: Miss Eugenie Elise Lurette is listed as the maid.
  Fact 0.1289: Miss Eugenie Elise Lurette is from New York City.
  Fact 0.1290: Miss Eugenie Elise Lurette boarded at Cherbourg.
  Fact 0.1291: Miss Eugenie Elise Lurette's destination is New York City.
  Fact 0.1292: Miss Eugenie Elise Lurette has 6 in the lifeboat or body column.
  Fact 0.1293: Stähelin-Maeglin, Dr. Max was 32 years old.
  Fact 0.1294: Stähelin-Maeglin, Dr. Max is from Basel, Switzerland.
  Fact 0.1295: Stähelin-Maeglin, Dr. Max boarded at Southampton.
  Fact 0.1296: Stähelin-Maeglin, Dr. Max's destination is New York City.
  Fact 0.1297: Stead, Mr. William Thomas was 62 years old.
  Fact 0.1298: Stead, Mr. William Thomas is from London, England, UK.
  Fact 0.1299: Stead, Mr. William Thomas boarded at Southampton.
  Fact 0.1300: Stead, Mr. William Thomas's destination is New York City.
  Fact 0.1301: Stengel, Mr. Charles Emil Henry was 54 years old.
  Fact 0.1302: Stengel, Mr. Charles Emil Henry is from Newark, New Jersey, US.
  Fact 0.1303: Stengel, Mr. Charles Emil Henry boarded at Cherbourg.
  Fact 0.1304: Stengel, Mr. Charles Emil Henry's destination is Newark, New Jersey, US.
  Fact 0.1305: Stengel, Mrs. Annie May (née Morris) was 44 years old.
  Fact 0.1306: Stengel, Mrs. Annie May (née Morris) is from Newark, New Jersey, US.
  Fact 0.1307: Stengel, Mrs. Annie May (née Morris) boarded at Cherbourg.
  Fact 0.1308: Stengel, Mrs. Annie May (née Morris)'s destination is Newark, New Jersey, US.
  Fact 0.1309: Stengel, Mrs. Annie May (née Morris) has 5 in the lifeboat or body column.
  Fact 0.1310: Stephenson, Mrs. Martha (née Eustis) was 52 years old.
  Fact 0.1311: Stephenson, Mrs. Martha (née Eustis) is from Paris, France.
  Fact 0.1312: Stephenson, Mrs. Martha (née Eustis) boarded at Cherbourg.
  Fact 0.1313: Stephenson, Mrs. Martha (née Eustis)'s destination is Haverford, Pennsylvania, US.
  Fact 0.1314: Stewart, Mr. Albert A. was 54 years old.
  Fact 0.1315: Stewart, Mr. Albert A. is from Cincinnati, Ohio, US.
  Fact 0.1316: Stewart, Mr. Albert A. boarded at Cherbourg.
  Fact 0.1317: Stewart, Mr. Albert A.'s destination is Gallipolis, Ohio, US.
  Fact 0.1318: Stone, Mrs. Martha Evelyn (née Stevens) was 62 years old.
  Fact 0.1319: Stone, Mrs. Martha Evelyn (née Stevens) is from New York City.
  Fact 0.1320: Stone, Mrs. Martha Evelyn (née Stevens) boarded at Southampton.
  Fact 0.1321: Stone, Mrs. Martha Evelyn (née Stevens)'s destination is New York City.
  Fact 0.1322: Stone, Mrs. Martha Evelyn (née Stevens) has 6 in the lifeboat or body column.
  Fact 0.1323: Miss Amelie "Amelia" Icard was 38 years old.
  Fact 0.1324: Miss Amelie "Amelia" Icard is listed as the maid.
  Fact 0.1325: Miss Amelie "Amelia" Icard is from New York City.
  Fact 0.1326: Miss Amelie "Amelia" Icard boarded at Southampton.
  Fact 0.1327: Miss Amelie "Amelia" Icard's destination is New York City.
  Fact 0.1328: Miss Amelie "Amelia" Icard has 6 in the lifeboat or body column.
  Fact 0.1329: Straus, Mr. Isidor was 67 years old.
  Fact 0.1330: Straus, Mr. Isidor is from New York City.
  Fact 0.1331: Straus, Mr. Isidor boarded at Southampton.
  Fact 0.1332: Straus, Mr. Isidor's destination is New York City.
  Fact 0.1333: Straus, Mr. Isidor has 96 in the lifeboat or body column.
  Fact 0.1334: Mr. John Farthing was 57 years old.
  Fact 0.1335: Mr. John Farthing is listed as the valet.
  Fact 0.1336: Mr. John Farthing is from New York City.
  Fact 0.1337: Mr. John Farthing boarded at Southampton.
  Fact 0.1338: Mr. John Farthing's destination is New York City.
  Fact 0.1339: Straus, Mrs. Rosalie Ida (née Blun) was 63 years old.
  Fact 0.1340: Straus, Mrs. Rosalie Ida (née Blun) is from New York City.
  Fact 0.1341: Straus, Mrs. Rosalie Ida (née Blun) boarded at Southampton.
  Fact 0.1342: Straus, Mrs. Rosalie Ida (née Blun)'s destination is New York City.
  Fact 0.1343: Miss Ellen Bird was 31 years old.
  Fact 0.1344: Miss Ellen Bird is listed as the maid.
  Fact 0.1345: Miss Ellen Bird is from New York City.
  Fact 0.1346: Miss Ellen Bird boarded at Southampton.
  Fact 0.1347: Miss Ellen Bird's destination is New York City.
  Fact 0.1348: Miss Ellen Bird has 8 in the lifeboat or body column.
  Fact 0.1349: Sutton, Mr. Frederick was 61 years old.
  Fact 0.1350: Sutton, Mr. Frederick is from Haddonfield, New Jersey, US.
  Fact 0.1351: Sutton, Mr. Frederick boarded at Southampton.
  Fact 0.1352: Sutton, Mr. Frederick's destination is Haddonfield, New Jersey, US.
  Fact 0.1353: Sutton, Mr. Frederick has 46 in the lifeboat or body column.
  Fact 0.1354: Swift, Mrs. Margaret Welles (née Barron) was 46 years old.
  Fact 0.1355: Swift, Mrs. Margaret Welles (née Barron) is from New York City.
  Fact 0.1356: Swift, Mrs. Margaret Welles (née Barron) boarded at Southampton.
  Fact 0.1357: Swift, Mrs. Margaret Welles (née Barron)'s destination is New York City.
  Fact 0.1358: Swift, Mrs. Margaret Welles (née Barron) has 8 in the lifeboat or body column.
  Fact 0.1359: Taussig, Mr. Emil was 52 years old.
  Fact 0.1360: Taussig, Mr. Emil is from New York City.
  Fact 0.1361: Taussig, Mr. Emil boarded at Southampton.
  Fact 0.1362: Taussig, Mr. Emil's destination is New York City.
  Fact 0.1363: Taussig, Mrs. Tillie (née Mandelbaum) was 39 years old.
  Fact 0.1364: Taussig, Mrs. Tillie (née Mandelbaum) is from New York City.
  Fact 0.1365: Taussig, Mrs. Tillie (née Mandelbaum) boarded at Southampton.
  Fact 0.1366: Taussig, Mrs. Tillie (née Mandelbaum)'s destination is New York City.
  Fact 0.1367: Taussig, Mrs. Tillie (née Mandelbaum) has 8 in the lifeboat or body column.
  Fact 0.1368: Taussig, Miss Ruth was 18 years old.
  Fact 0.1369: Taussig, Miss Ruth is from New York City.
  Fact 0.1370: Taussig, Miss Ruth boarded at Southampton.
  Fact 0.1371: Taussig, Miss Ruth's destination is New York City.
  Fact 0.1372: Taussig, Miss Ruth has 8 in the lifeboat or body column.
  Fact 0.1373: Taylor, Mr. Elmer Zebley was 48 years old.
  Fact 0.1374: Taylor, Mr. Elmer Zebley is from London, England, UK.
  Fact 0.1375: Taylor, Mr. Elmer Zebley boarded at Southampton.
  Fact 0.1376: Taylor, Mr. Elmer Zebley's destination is East Orange, New Jersey, US.
  Fact 0.1377: Taylor, Mr. Elmer Zebley has 5 in the lifeboat or body column.
  Fact 0.1378: Taylor, Mrs. Juliet Cummins (née Wright) was 48 years old.
  Fact 0.1379: Taylor, Mrs. Juliet Cummins (née Wright) is from London, England, UK.
  Fact 0.1380: Taylor, Mrs. Juliet Cummins (née Wright) boarded at Southampton.
  Fact 0.1381: Taylor, Mrs. Juliet Cummins (née Wright)'s destination is East Orange, New Jersey, US.
  Fact 0.1382: Taylor, Mrs. Juliet Cummins (née Wright) has 5 in the lifeboat or body column.
  Fact 0.1383: Thayer, Mr. John Borland II was 49 years old.
  Fact 0.1384: Thayer, Mr. John Borland II is from Haverford, Pennsylvania, US.
  Fact 0.1385: Thayer, Mr. John Borland II boarded at Cherbourg.
  Fact 0.1386: Thayer, Mr. John Borland II's destination is Haverford, Pennsylvania, US.
  Fact 0.1387: Thayer, Mrs. Marian Longsteth (née Morris) was 39 years old.
  Fact 0.1388: Thayer, Mrs. Marian Longsteth (née Morris) is from Haverford, Pennsylvania, US.
  Fact 0.1389: Thayer, Mrs. Marian Longsteth (née Morris) boarded at Cherbourg.
  Fact 0.1390: Thayer, Mrs. Marian Longsteth (née Morris)'s destination is Haverford, Pennsylvania, US.
  Fact 0.1391: Miss Margaret Fleming was 42 years old.
  Fact 0.1392: Miss Margaret Fleming is listed as the maid.
  Fact 0.1393: Miss Margaret Fleming is from Haverford, Pennsylvania, US.
  Fact 0.1394: Miss Margaret Fleming boarded at Cherbourg.
  Fact 0.1395: Miss Margaret Fleming's destination is Haverford, Pennsylvania, US.
  Fact 0.1396: Thayer, Mr. John Borland "Jack" III was 17 years old.
  Fact 0.1397: Thayer, Mr. John Borland "Jack" III is from Haverford, Pennsylvania, US.
  Fact 0.1398: Thayer, Mr. John Borland "Jack" III boarded at Cherbourg.
  Fact 0.1399: Thayer, Mr. John Borland "Jack" III's destination is Haverford, Pennsylvania, US.
  Fact 0.1400: Thorne, Mrs. Gertrude Maybelle (née McMinn) was 37 years old.
  Fact 0.1401: Thorne, Mrs. Gertrude Maybelle (née McMinn) is from New York City, New York, US.
  Fact 0.1402: Thorne, Mrs. Gertrude Maybelle (née McMinn) boarded at Cherbourg.
  Fact 0.1403: Thorne, Mrs. Gertrude Maybelle (née McMinn)'s destination is New York City, New York, US.
  Fact 0.1404: Tucker, Mr. Gilbert Milligan Jr. was 31 years old.
  Fact 0.1405: Tucker, Mr. Gilbert Milligan Jr. is from Albany, New York, US.
  Fact 0.1406: Tucker, Mr. Gilbert Milligan Jr. boarded at Cherbourg.
  Fact 0.1407: Tucker, Mr. Gilbert Milligan Jr.'s destination is Albany, New York, US.
  Fact 0.1408: Tucker, Mr. Gilbert Milligan Jr. has 7 in the lifeboat or body column.
  Fact 0.1409: Uruchurtu, Don Manuel E. was 40 years old.
  Fact 0.1410: Uruchurtu, Don Manuel E. is from Mexico City, Mexico.
  Fact 0.1411: Uruchurtu, Don Manuel E. boarded at Cherbourg.
  Fact 0.1412: Uruchurtu, Don Manuel E.'s destination is Mexico City, Mexico.
  Fact 0.1413: Van der Hoef, Mr. Wyckoff was 61 years old.
  Fact 0.1414: Van der Hoef, Mr. Wyckoff is from Brooklyn, New York, US.
  Fact 0.1415: Van der Hoef, Mr. Wyckoff boarded at Belfast.
  Fact 0.1416: Van der Hoef, Mr. Wyckoff's destination is Brooklyn, New York, US.
  Fact 0.1417: Van der Hoef, Mr. Wyckoff has 245 in the lifeboat or body column.
  Fact 0.1418: Walker, Mr. William Anderson was 48 years old.
  Fact 0.1419: Walker, Mr. William Anderson is from East Orange, New Jersey, US.
  Fact 0.1420: Walker, Mr. William Anderson boarded at Southampton.
  Fact 0.1421: Walker, Mr. William Anderson's destination is East Orange, New Jersey, US.
  Fact 0.1422: Warren, Mr. Frank Manley was 63 years old.
  Fact 0.1423: Warren, Mr. Frank Manley is from Portland, Oregon, US.
  Fact 0.1424: Warren, Mr. Frank Manley boarded at Cherbourg.
  Fact 0.1425: Warren, Mr. Frank Manley's destination is Portland, Oregon, US.
  Fact 0.1426: Warren, Mrs. Anna Sophia (née Atkinson) was 60 years old.
  Fact 0.1427: Warren, Mrs. Anna Sophia (née Atkinson) is from Portland, Oregon, US.
  Fact 0.1428: Warren, Mrs. Anna Sophia (née Atkinson) boarded at Cherbourg.
  Fact 0.1429: Warren, Mrs. Anna Sophia (née Atkinson)'s destination is Portland, Oregon, US.
  Fact 0.1430: Warren, Mrs. Anna Sophia (née Atkinson) has 5 in the lifeboat or body column.
  Fact 0.1431: Weir, Colonel John was 59 years old.
  Fact 0.1432: Weir, Colonel John is from New York City.
  Fact 0.1433: Weir, Colonel John boarded at Southampton.
  Fact 0.1434: Weir, Colonel John's destination is New York City.
  Fact 0.1435: White, Mr. Percival Wayland was 54 years old.
  Fact 0.1436: White, Mr. Percival Wayland is from Brunswick, Maine, US.
  Fact 0.1437: White, Mr. Percival Wayland boarded at Southampton.
  Fact 0.1438: White, Mr. Percival Wayland's destination is Brunswick, Maine, US.
  Fact 0.1439: White, Mr. Richard Frasar was 21 years old.
  Fact 0.1440: White, Mr. Richard Frasar is from Brunswick, Maine, US.
  Fact 0.1441: White, Mr. Richard Frasar boarded at Southampton.
  Fact 0.1442: White, Mr. Richard Frasar's destination is Brunswick, Maine, US.
  Fact 0.1443: White, Mr. Richard Frasar has 169 in the lifeboat or body column.
  Fact 0.1444: White, Mrs. Ella (née Holmes) was 55 years old.
  Fact 0.1445: White, Mrs. Ella (née Holmes) is from New York City.
  Fact 0.1446: White, Mrs. Ella (née Holmes) boarded at Cherbourg.
  Fact 0.1447: White, Mrs. Ella (née Holmes)'s destination is New York City.
  Fact 0.1448: White, Mrs. Ella (née Holmes) has 8 in the lifeboat or body column.
  Fact 0.1449: Miss Amelia Mayo "Nellie" Bessette was 39 years old.
  Fact 0.1450: Miss Amelia Mayo "Nellie" Bessette is listed as the maid.
  Fact 0.1451: Miss Amelia Mayo "Nellie" Bessette is from New York City.
  Fact 0.1452: Miss Amelia Mayo "Nellie" Bessette boarded at Cherbourg.
  Fact 0.1453: Miss Amelia Mayo "Nellie" Bessette's destination is New York City.
  Fact 0.1454: Miss Amelia Mayo "Nellie" Bessette has 8 in the lifeboat or body column.
  Fact 0.1455: Mr. Sante Righini was 22 years old.
  Fact 0.1456: Mr. Sante Righini is listed as the manservant.
  Fact 0.1457: Mr. Sante Righini is from New York City.
  Fact 0.1458: Mr. Sante Righini boarded at Cherbourg.
  Fact 0.1459: Mr. Sante Righini's destination is New York City.
  Fact 0.1460: Mr. Sante Righini has 232 in the lifeboat or body column.
  Fact 0.1461: Wick, Colonel George Dennick was 58 years old.
  Fact 0.1462: Wick, Colonel George Dennick is from Youngstown, Ohio, US.
  Fact 0.1463: Wick, Colonel George Dennick boarded at Southampton.
  Fact 0.1464: Wick, Colonel George Dennick's destination is Youngstown, Ohio, US.
  Fact 0.1465: Wick, Colonel George Dennick has 8 in the lifeboat or body column.
  Fact 0.1466: Wick, Mrs. Mary (née Hitchcock) was 45 years old.
  Fact 0.1467: Wick, Mrs. Mary (née Hitchcock) is from Youngstown, Ohio, US.
  Fact 0.1468: Wick, Mrs. Mary (née Hitchcock) boarded at Southampton.
  Fact 0.1469: Wick, Mrs. Mary (née Hitchcock)'s destination is Youngstown, Ohio, US.
  Fact 0.1470: Wick, Mrs. Mary (née Hitchcock) has 8 in the lifeboat or body column.
  Fact 0.1471: Wick, Miss Mary Natalie was 31 years old.
  Fact 0.1472: Wick, Miss Mary Natalie is from Youngstown, Ohio, US.
  Fact 0.1473: Wick, Miss Mary Natalie boarded at Southampton.
  Fact 0.1474: Wick, Miss Mary Natalie's destination is Youngstown, Ohio, US.
  Fact 0.1475: Wick, Miss Mary Natalie has 8 in the lifeboat or body column.
  Fact 0.1476: Widener, Mr. George Dunton was 50 years old.
  Fact 0.1477: Widener, Mr. George Dunton is from Philadelphia, Pennsylvania, US.
  Fact 0.1478: Widener, Mr. George Dunton boarded at Southampton.
  Fact 0.1479: Widener, Mr. George Dunton's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1480: Mr. Edwin Herbert Keeping was 33 years old.
  Fact 0.1481: Mr. Edwin Herbert Keeping is listed as the valet.
  Fact 0.1482: Mr. Edwin Herbert Keeping is from Philadelphia, Pennsylvania, US.
  Fact 0.1483: Mr. Edwin Herbert Keeping boarded at Southampton.
  Fact 0.1484: Mr. Edwin Herbert Keeping's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1485: Mr. Edwin Herbert Keeping has 45 in the lifeboat or body column.
  Fact 0.1486: Widener, Mrs. Eleanor (née Elkins) was 50 years old.
  Fact 0.1487: Widener, Mrs. Eleanor (née Elkins) is from Philadelphia, Pennsylvania, US.
  Fact 0.1488: Widener, Mrs. Eleanor (née Elkins) boarded at Southampton.
  Fact 0.1489: Widener, Mrs. Eleanor (née Elkins)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.1490: Miss Amalie Henriette "Emily" Gieger was 35 years old.
  Fact 0.1491: Miss Amalie Henriette "Emily" Gieger is listed as the maid.
  Fact 0.1492: Miss Amalie Henriette "Emily" Gieger is from Philadelphia, Pennsylvania, US.
  Fact 0.1493: Miss Amalie Henriette "Emily" Gieger boarded at Southampton.
  Fact 0.1494: Miss Amalie Henriette "Emily" Gieger's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1495: Widener, Mr. Harry Elkins was 27 years old.
  Fact 0.1496: Widener, Mr. Harry Elkins is from Philadelphia, Pennsylvania, US.
  Fact 0.1497: Widener, Mr. Harry Elkins boarded at Southampton.
  Fact 0.1498: Widener, Mr. Harry Elkins's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1499: Willard, Miss Constance was 21 years old.
  Fact 0.1500: Willard, Miss Constance is from Duluth, Minnesota, US.
  Fact 0.1501: Willard, Miss Constance boarded at Southampton.
  Fact 0.1502: Willard, Miss Constance's destination is Duluth, Minnesota, US.
  Fact 0.1503: Willard, Miss Constance has 8 in the lifeboat or body column.
  Fact 0.1504: Williams, Mr. Charles Duane was 51 years old.
  Fact 0.1505: Williams, Mr. Charles Duane is from Geneva, Switzerland.
  Fact 0.1506: Williams, Mr. Charles Duane boarded at Cherbourg.
  Fact 0.1507: Williams, Mr. Charles Duane's destination is Radnor, Pennsylvania, US.
  Fact 0.1508: Williams, Mr. Richard Norris II was 21 years old.
  Fact 0.1509: Williams, Mr. Richard Norris II is from Geneva, Switzerland.
  Fact 0.1510: Williams, Mr. Richard Norris II boarded at Cherbourg.
  Fact 0.1511: Williams, Mr. Richard Norris II's destination is Radnor, Pennsylvania, US.
  Fact 0.1512: Woolner, Mr. Hugh was 45 years old.
  Fact 0.1513: Woolner, Mr. Hugh is from London, England, UK.
  Fact 0.1514: Woolner, Mr. Hugh boarded at Southampton.
  Fact 0.1515: Woolner, Mr. Hugh's destination is New York City.
  Fact 0.1516: Wright, Mr. George was 62 years old.
  Fact 0.1517: Wright, Mr. George is from Halifax, Nova Scotia, Canada.
  Fact 0.1518: Wright, Mr. George boarded at Southampton.
  Fact 0.1519: Wright, Mr. George's destination is Halifax, Nova Scotia, Canada.
  Fact 0.1520: Young, Miss Marie Grice was 36 years old.
  Fact 0.1521: Young, Miss Marie Grice is from New York City.
  Fact 0.1522: Young, Miss Marie Grice boarded at Cherbourg.
  Fact 0.1523: Young, Miss Marie Grice's destination is New York City.
  Fact 0.1524: Young, Miss Marie Grice has 8 in the lifeboat or body column.

Retrieved items:
<item 1>
Passengers of the RMS Titanic
Frank M. Warren , Sr . George Dennick Wick George Dunton Widener Harry Elkins Widener Duane Williams George Henry Wright Roderick Chisholm Survivors ( last
living ) Rhoda Abbott Trevor Allison Lillian Asplund Madeleine Astor Ruth Becker Lawrence Beesley Karl Behr Dickinson Bishop Mauritz Håkan Björnström -
Steffansson Elsie Bowerman Francis Browne Margaret `` Molly '' Brown Daniel Buckley Alden Caldwell Helen Churchill Candee Charlotte Drake Cardeza Lucile Carter
Gladys Cherry Millvina Dean Margaret Devaney Sir Cosmo Duff - Gordon Lucy , Lady Duff - Gordon Ethel Flora Fortune Dorothy Gibson Archibald Gracie IV Frank John
William Goldsmith Edith Haisman Henry S. Harper Eva Hart Margaret Bechstein Hays Masabumi Hosono J. Bruce Ismay Eleanor Ileen Johnson Louise Laroche Louise Kink
Margaret Mannion Michel Marcel Navratil Alfred Nourney Arthur Godfrey Peuchen Winnifred Quick Marjorie Newell Robb Edith Rosenbaum Noël Leslie ,
</item 1>

<item 2>
Graph query row:
person: Lawrence Beesley
relationship: RELATED_TO
connected_entity: RMS Titanic
role: None
fact: None
result: None
outcome: None
</item 2>

<item 3>
Passengers of the RMS Titanic
Jump up ^ Brewster , Hugh ; Coulter , Laurie ( 1998 ) . 882 1 / 2 Amazing Answers To Your Questions About The Titanic . Scholastic Paperbacks . p. 18 . ISBN 978
- 0 - 8117 - 1814 - 1 . Jump up ^ I Survived the Titanic : Ruth Becker Jump up ^ `` Whitaker Centre -- ' ' Titanic Artifact Exhibit : Ministry Through Tragedy '
' '' ( PDF ) . Archived from the original ( PDF ) on 4 July 2011 . Retrieved 6 July 2011 . Jump up ^ `` Fr Thomas Roussel Davids Byles '' . Encyclopedia
Titanica . Retrieved 15 July 2011 . Jump up ^ `` Pastor John Harper and the Titanic Story '' . Harper Memorial Baptist Church . Archived from the original on 17
April 2012 . Retrieved 2 April 2012 . Jump up ^ `` Lawrence Beesley -- Celebrated Titanic Survivor '' . Bygonederbyshire.co.uk . Retrieved 6 July 2011 .
</item 3>

<item 4>
Passengers of the RMS Titanic
ISBN 978 - 0 - 14 - 004757 - 8 . Jump up ^ `` Passenger List and Survivors of Steamship Titanic '' . United States Senate Inquiry. 30 July 1912 . Retrieved 15
July 2011 . Jump up ^ Hall , Wayne ( 1986 ) . `` Social Class and Survival on the SS Titanic '' ( PDF ) . Social Science & Medicine . 22 ( 6 ) : 687 -- 690 .
Jump up ^ Barratt , Nick ( 2009 ) . Lost Voices From the Titanic : The Definitive Oral History . London : Random House . p. 93 . ISBN 978 - 1 - 84809 - 151 - 1
. Jump up ^ Howells , Richard ( 1999 ) . The Myth of the Titanic . United Kingdom : MacMillan Press . p. 18 . ISBN 978 - 0 - 333 - 72597 - 9 .
</item 4>

<item 5>
Passengers of the RMS Titanic
Although several people managed to reach this lifeboat , three died during the night . When Fifth Officer Harold Lowe and six crewmen returned to the wreck site
after the sinking with an empty lifeboat to pick up survivors , they rescued surviving passengers from Collapsible A , but left the three dead bodies in the
boat : Thomson Beattie , a first - class passenger , and two crew members , a fireman and a seaman . After their retrieval from Collapsible A by Oceanic , the
bodies were buried at sea . Passenger list ( edit ) Breakdown of casualties according to the British Board of Trade report ( click for detail ) The following is
a full list of known passengers who sailed on the maiden voyage of the RMS Titanic . Included in this list are the nine - member Guarantee Group and the eight
members of the ship 's band , listed as both passengers and crew . They are also included in the list of crew members on board RMS Titanic . Passengers are
colour - coded , indicating whether they were saved or perished . The passenger did not survive The passenger survived Survivors are listed with the lifeboat
from which they were known to be rescued .
</item 5>

<item 6>
Graph query row:
person: Ruth Becker
relationship: RELATED_TO
connected_entity: RMS Titanic
role: None
fact: None
result: None
outcome: None
</item 6>

<item 7>
Passengers of the RMS Titanic
( Halifax ) Related Ships RMS Olympic RMS Carpathia HMHS Britannic CS Mackay - Bennett SS Californian Replica Titanic Titanic II Others White Star Line Titanic
Historical Society Encyclopedia Titanica Halomonas titanicae Women and children first SOS Robert Ballard Last remaining survivors of the RMS Titanic disaster
2000 -- 2009 Millvina Dean ( 1912 -- 2009 ) Barbara West ( 1911 -- 2007 ) Lillian Asplund ( 1906 -- 2006 ) Winnifred Quick ( 1904 -- 2002 ) Michel Navratil (
1908 -- 2001 ) 1995 -- 1999 Eleanor Johnson ( 1910 -- 1998 ) Louise Laroche ( 1910 -- 1998 ) Edith Brown ( 1896 -- 1997 ) Eva Hart ( 1905 -- 1996 ) Beatrice
Sandström ( 1910 -- 1995 ) 1990 -- 1994 Robertha `` Bertha '' Watt ( 1899 -- 1993 ) Ellen Shine ( 1891 -- 1993 ) Marjorie Newell ( 1889 -- 1992 ) Louise
</item 7>

<item 8>
Graph query row:
person: Millvina Dean
relationship: RELATED_TO
connected_entity: RMS Titanic
role: None
fact: None
result: None
outcome: None
</item 8>

<item 9>
Passengers of the RMS Titanic
Victims whose remains were recovered after the sinking are listed with a superscript next to the body number , indicating the recovery vessel : MB -- CS Mackay
- Bennett ( bodies 1 -- 306 ) M -- CS Minia ( bodies 307 -- 323 ) MM -- CGS Montmagny ( bodies 326 -- 329 ) A -- SS Algerine ( body 330 ) O -- RMS Oceanic (
bodies 331 -- 333 ) I -- SS Ilford ( body 334 ) OT -- SS Ottawa ( body 335 ) Numbers 324 and 325 were unused , and the six bodies buried at sea by the Carpathia
also went unnumbered . First class ( edit ) Name Age Hometown Boarded Destination Lifeboat Body Allen , Miss Elizabeth Walton 29 St. Louis , Missouri , US
Southampton St. Louis , Missouri , US Allison , Mr. Hudson Joshua Creighton 30 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 135 and
chauffeur , Mr.
</item 9>

<item 10>
Passengers of the RMS Titanic
Of the 711 passengers and crew rescued by the Carpathia , six , including first class passenger William F. Hoyt , either died in a lifeboat during the night or
on board the Carpathia the next morning , and were buried at sea . In the days following the sinking , several ships sailed to the disaster area to recover
victims ' bodies . The White Star Line chartered the cable ship Mackay - Bennett from Halifax , Nova Scotia to retrieve bodies . Three other ships followed in
the search : the cable ship Minia , the lighthouse supply ship Montmagny and the sealing vessel Algerine . Each ship left with embalming supplies , undertakers
, and clergy . Upon recovery , each body retrieved by the Mackay - Bennett was numbered and given as detailed a description as possible to help aid in
identification . The physical appearance of each body -- height , weight , age , hair and eye colour , visible birthmarks , scars or tattoos , was catalogued
and any personal effects on the bodies were gathered and placed in small canvas bags corresponding to their number .
</item 10>
=== END ITEM id=5e138247e8b69b3d ===

=== ITEM id=d794f5ef2b5ac841 ===
Gold sentences and facts:
Sentence 0: A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called `` perfecting the security interest '' in the property , and this type of loan is a secured loan . A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit documents and the lease , respectively , and not the financing statement .
  Fact 0.0: A UCC-1 financing statement is a legal form.
  Fact 0.1: UCC-1 is an abbreviation for Uniform Commercial Code-1.
  Fact 0.2: A creditor files a UCC-1 financing statement to give notice that it has or may have an interest in the personal property of a debtor.
  Fact 0.3: A debtor is a person who owes a debt to the creditor, as typically specified in the agreement creating the debt.
  Fact 0.4: A financing statement is filed to perfect a creditor's security interest.
  Fact 0.5: The filing gives public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt, with a certain priority.
  Fact 0.6: Such notices of sale are often found in local newspapers.
  Fact 0.7: Once the form has been filed, the creditor establishes a relative priority with other creditors of the debtor.
  Fact 0.8: Perfecting the security interest is also called perfecting the security interest in the property.
  Fact 0.9: This type of loan is a secured loan.
  Fact 0.10: A financing statement may also be filed in the real estate records by a lessor of fixtures.
  Fact 0.11: The real estate filing establishes the lessor's priority against a holder of a mortgage or other lien on the real property.
  Fact 0.12: The creditor's rights against the debtor are based on the credit documents, not the financing statement.
  Fact 0.13: The lessor's rights against the lessee are based on the lease, not the financing statement.
Sentence 1: The financing statement is generally filed with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which is tasked with overseeing business organizations and activities , including receipt of financing statements . However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures . In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where third parties are most likely to search for such record .
  Fact 1.0: The financing statement is generally filed with the office of the state secretary of state.
  Fact 1.1: The financing statement is generally filed in the state where the debtor is located.
  Fact 1.2: For an individual, the state where the debtor is located is the state where the debtor resides.
  Fact 1.3: For most kinds of business organizations, the state where the debtor is located is the state of incorporation or organization.
  Fact 1.4: Many states have a state agency that operates under the secretary of state.
  Fact 1.5: That state agency is tasked with overseeing business organizations and activities, including receipt of financing statements.
  Fact 1.6: An exception exists if the collateral is tied to a particular piece of real property, such as timber, mineral rights, or fixtures.
  Fact 1.7: In that exception case, the filing must be made in the county where the property is located.
  Fact 1.8: In that exception case, the filing is usually made in the recording office or county court.
  Fact 1.9: Third parties are most likely to search for such a record in that county.

Retrieved items:
<item 1>
UCC-1 financing statement
However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures
. In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where
third parties are most likely to search for such record . In the case of a loan secured by personal property collateral , the filing of a financing statement
gives notice of a lien against the property so that other lenders or buyers of the personal property will know of the security interest . In the case of a
filing of a financing statement by a lessor of fixtures , the filing of the financing statement gives notice of the lessor 's interests to others who acquire an
interest in the real property and related fixtures . The financing statement does not create a lien nor does it create any additional rights against a lessee in
favor of a lessor , the filing of a financing statement just gives notice of whatever rights the creditor or lessor have under their loan documents or lease ,
respectively .
</item 1>

<item 2>
IRS penalties
U.S. corporations more than 25 % owned , directly or indirectly , by foreign persons must file Form 5472 to report such ownership and all transactions with
related parties . Failure to timely file carries a $10,000 penalty per required form . This penalty may be increased by $10,000 per month per form for continued
failure to file . In addition , taxpayers who fail to report changes in foreign taxes used as credits against Federal income tax may be subject to penalties .
U.S. citizen or resident taxpayers ( including entities ) who are beneficiaries of a foreign trust or make transfers of property to a foreign trust must report
information about the transfer and the trust or corporation . Failure to timely report on Form 3520 or Form 3520 - A may result in penalties of up to 35 % .
Similar transferors to foreign corporations failing to file Form 926 may face penalties of 10 % of the value of the transfer , up to $100,000 . Excise taxes as
penalties ( edit ) Federal excise taxes are imposed on a variety of goods and services . Some of these taxes require purchase of tax stamps or other evidence of
advance payment of tax .
</item 2>

<item 3>
UCC-1 financing statement
UCC - 1 financing statement - wikipedia UCC - 1 financing statement Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2014 ) ( Learn how and
when to remove this template message ) A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files
to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the
agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to
take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local
newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called ``
perfecting the security interest '' in the property , and this type of loan is a secured loan .
</item 3>

<item 4>
Ownership
A structured group duly constituted as an entity under law may still not protect members from being personally liable for each other 's actions . Court
decisions against the entity itself may give rise to unlimited personal liability for each and every member . An example of this situation is a professional
partnership ( e.g. law practice ) in some jurisdictions . Thus , being a partner or owner in a group may give little advantage in terms of share ownership while
producing a lot of risk to the partner , owner or participant . Sharing gains ( edit ) At the end of each financial year , accounting rules determine a surplus
or profit , which may be retained inside the entity or distributed among owners according to the initial setup intent when the entity was created . For public
corporations , common shareholders have no right to receive any of the profit . Entities with a member focus will give financial surplus back to members
according to the volume of financial activity that the participating member generated for the entity . Examples of this are producer cooperatives , buyer
cooperatives and participating whole life policyholders in both mutual and share - capital insurance companies . Entities with share voting rights that depend
on financial capital distribute surplus among shareholders without regard to any other contribution to the entity .
</item 4>

<item 5>
Ownership
Types of owners ( edit ) In person ( edit ) Individuals may own property directly . In some societies only adult men may own property ; in other societies (
such as the Haudenosaunee ) , property is matrilinear and passed on from mother to the offspring . In most societies both men and women can own property with no
restrictions and limitations at all . Structured ownership entities ( edit ) Throughout history , nations ( or governments ) and religious organizations have
owned property . These entities exist primarily for other purposes than to own or operate property ; hence , they may have no clear rules regarding the
disposition of their property . To own and operate property , structures ( often known today as legal entities ) have been created in many societies throughout
history . The differences in how they deal with members ' rights is a key factor in determining their type . Each type has advantages and disadvantages derived
from their means of recognizing or disregarding ( rewarding or not ) contributions of financial capital or personal effort . Cooperatives , corporations ,
trusts , partnerships , and condominium associations are only some of the many varied types of structured ownership ; each type has many subtypes .
</item 5>

<item 6>
UCC-1 financing statement
A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of
a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit
documents and the lease , respectively , and not the financing statement . Pursuant to the standards set forth in the UCC , at 9 - 503 and 9 - 504 , the
financing statement need only contain three pieces of information : the debtor 's name and address the creditor 's name and address an indication of the
collateral , `` whether or not it is specific , if it reasonably identifies what is described . '' ( UCC 9 - 108 ) The financing statement is generally filed
with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most
kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which
is tasked with overseeing business organizations and activities , including receipt of financing statements .
</item 6>

<item 7>
Board of directors
Such agents have duties to discharge of a fiduciary nature towards their principal . And it is a rule of universal application that no one , having such duties
to discharge , shall be allowed to enter into engagements in which he has , or can have , a personal interest conflicting or which possibly may conflict , with
the interests of those whom he is bound to protect ... So strictly is this principle adhered to that no question is allowed to be raised as to the fairness or
unfairness of the contract entered into ... '' ( emphasis added ) However , in many jurisdictions the members of the company are permitted to ratify
transactions which would otherwise fall foul of this principle . It is also largely accepted in most jurisdictions that this principle can be overridden in the
company 's constitution . In many countries , there is also a statutory duty to declare interests in relation to any transactions , and the director can be
fined for failing to make disclosure . Use of corporate property , opportunity , or information ( edit ) Directors must not , without the informed consent of
the company , use for their own profit the company 's assets , opportunities , or information .
</item 7>

<item 8>
UCC-1 financing statement
References ( edit ) Jump up ^ http://www.findlegalforms.com/articles/form-encyclopedia/what-is-a-ucc1-financing-statement Jump up ^
http://biztaxlaw.about.com/od/glossaryu/g/ucc1statement.htm Retrieved from ``
https://en.wikipedia.org/w/index.php?title=UCC-1_financing_statement&oldid=803697764 '' Categories : Uniform Commercial Code Legal terminology Hidden categories
: Articles needing additional references from October 2014 All articles needing additional references Talk About Wikipedia Add links This page was last edited
on 4 October 2017 , at 02 : 00 . About Wikipedia
</item 8>

<item 9>
Board of directors
Jump up ^ This division was rejected in British Columbia in Teck Corporation v Millar ( 1972 ) 33 DLR ( 3d ) 288 . Jump up ^ Although as Gower points out , as
well understood as the rule is , there is a paucity of authority on the point . But see Clark v Workman ( 1920 ) 1 Ir R 107 and Dawson International plc v Coats
Paton plc 1989 SLT 655 . Jump up ^ In the United Kingdom , see section 317 of the Companies Act 1985 . Jump up ^ In summary , the facts were as follows :
Company A owned a cinema , and the directors decided to acquire two other cinemas with a view to selling the entire undertaking as a going concern . They formed
a new company ( `` Company B '' ) to take the leases of the two new cinemas . But the lessor insisted on various stipulations , one of which was that Company B
had to have a paid up share capital of not less than £ 5,000 ( a substantial sum at the time ) .
</item 9>

<item 10>
Ownership
A legal entity is a legal construct through which the law allows a group of natural persons to act as if it were an individual for certain purposes . Some duly
incorporated entities may not be owned by individuals nor by other entities ; they exist without being owned once they are created . Not being owned , they can
not be bought and sold . Mutual life insurance companies , credit unions , foundations and cooperatives , not for profit organizations , and public corporations
are examples of this . No person can purchase the company , as their ownership is not legally available for sale , neither as shares nor as a single whole .
Intellectual property ( edit ) Main article : Intellectual property Intellectual property ( IP ) refers to a legal entitlement which sometimes attaches to the
expressed form of an idea , or to some other intangible subject matter . This legal entitlement generally enables its holder to exercise exclusive rights of use
in relation to the subject matter of the IP . The term intellectual property reflects the idea that this subject matter is the product of the mind or the
intellect , and that IP rights may be protected at law in the same way as any other form of property .
</item 10>
=== END ITEM id=d794f5ef2b5ac841 ===

=== ITEM id=822a8ef2bdcd52d3 ===
Gold sentences and facts:
Sentence 0: A board of directors is a recognized group of people who jointly oversee the activities of an organization , which can be either a for - profit business , nonprofit organization , or a government agency . Such a board 's powers , duties , and responsibilities are determined by government regulations ( including the jurisdiction 's corporations law ) and the organization 's own constitution and bylaws . These authorities may specify the number of members of the board , how they are to be chosen , and how often they are to meet .
  Fact 0.0: A board of directors is a recognized group of people.
  Fact 0.1: A board of directors jointly oversees the activities of an organization.
  Fact 0.2: An organization overseen by a board of directors can be a for-profit business, a nonprofit organization, or a government agency.
  Fact 0.3: A board's powers, duties, and responsibilities are determined by government regulations, including the jurisdiction's corporations law.
  Fact 0.4: A board's powers, duties, and responsibilities are determined by the organization's own constitution and bylaws.
  Fact 0.5: The authorities over a board may specify the number of members of the board.
  Fact 0.6: The authorities over a board may specify how board members are chosen.
  Fact 0.7: The authorities over a board may specify how often the board is to meet.

Retrieved items:
<item 1>
Board of directors
Process and structure ( edit ) The process for running a board , sometimes called the board process , includes the selection of board members , the setting of
clear board objectives , the dissemination of documents or board package to the board members , the collaborative creation of an agenda for the meeting , the
creation and follow - up of assigned action items , and the assessment of the board process through standardized assessments of board members , owners , and
CEOs . The science of this process has been slow to develop due to the secretive nature of the way most companies run their boards , however some
standardization is beginning to develop . Some who are pushing for this standardization in the USA are the National Association of Corporate Directors ,
McKinsey Consulting and The Board Group . Board meetings ( edit ) A board of directors conducts its meetings according to the rules and procedures contained in
its governing documents . These procedures may allow the board to conduct its business by conference call or other electronic means . They may also specify how
a quorum is to be determined . Most organizations have adopted Robert 's Rules of Order as its guide to supplement its own rules .
</item 1>

<item 2>
Board of directors
The board of directors appoints the chief executive officer of the corporation and sets out the overall strategic direction . In corporations with dispersed
ownership , the identification and nomination of directors ( that shareholders vote for or against ) are often done by the board itself , leading to a high
degree of self - perpetuation . In a non-stock corporation with no general voting membership , the board is the supreme governing body of the institution ; its
members are sometimes chosen by the board itself . Contents ( hide ) 1 Terminology 2 Roles 3 Directors 3.1 Inside director 3.2 Outside director 3.3 Terminology
4 Process and structure 4.1 Board meetings 4.2 Size 5 Non-corporate boards 5.1 Membership organizations 6 Corporations 6.1 Governance 6.2 Two - tier system 6.3
History 6.4 Election and removal 6.5 Exercise of powers 6.6 Duties 6.6. 1 `` Proper purpose '' 6.6. 2 `` Unfettered discretion '' 6.6. 3 `` Conflict of duty and
interest '' 6.6.
</item 2>

<item 3>
Board of directors
Board of directors - wikipedia Board of directors Jump to : navigation , search `` Board Room '' and `` Board of Trustees '' redirect here . For other uses ,
see Board Room ( disambiguation ) and Board of Trustees ( disambiguation ) . A meeting of a board of directors of the Leipzig -- Dresden Railway Company in 1852
A board of directors is a recognized group of people who jointly oversee the activities of an organization , which can be either a for - profit business ,
nonprofit organization , or a government agency . Such a board 's powers , duties , and responsibilities are determined by government regulations ( including
the jurisdiction 's corporations law ) and the organization 's own constitution and bylaws . These authorities may specify the number of members of the board ,
how they are to be chosen , and how often they are to meet . In an organization with voting members , the board is accountable to , and might be subordinate to
, the organization 's full membership , which usually vote for the members of the board . In a stock corporation , non-executive directors are voted for by the
shareholders and the board is the highest authority in the management of the corporation .
</item 3>

<item 4>
Board of directors
The setup of a board of directors vary widely across organizations and may include provisions that are applicable to corporations , in which the `` shareholders
'' are the members of the organization . A difference may be that the membership elects the officers of the organization , such as the president and the
secretary , and the officers become members of the board in addition to the directors and retain those duties on the board . The directors may also be
classified as officers in this situation . There may also be ex-officio members of the board , or persons who are members due to another position that they hold
. These ex-officio members have all the same rights as the other board members . Members of the board may be removed before their term is complete . Details on
how they can be removed are usually provided in the bylaws . If the bylaws do not contain such details , the section on disciplinary procedures in Robert 's
Rules of Order may be used . Corporations ( edit ) In a publicly held company , directors are elected to represent and are legally obligated as fiduciaries to
represent owners of the company -- the shareholders / stockholders .
</item 4>

<item 5>
Board of directors
3 `` Conflict of duty and interest '' 6.6. 3.1 Transactions with the company 6.6. 3.2 Use of corporate property , opportunity , or information 6.6. 3.3
Competing with the company 6.6. 4 Common law duties of care and skill 6.6. 5 Remedies for breach of duty 6.6. 6 Current trends 6.7 United States 6.7. 1 Sarbanes
-- Oxley Act 6.7. 2 Size 6.7. 3 Committees 6.7. 4 Compensation 6.8 Criticism 7 See also 8 Notes 9 References 9.1 Citations 9.2 Sources 10 External links
Terminology ( edit ) Other names include board of directors and advisors , board of governors , board of managers , board of regents , board of trustees , or
board of visitors . It may also be called `` the executive board '' and is often simply referred to as `` the board '' . Roles ( edit ) Business administration
Management of a business Accounting ( show ) Management accounting Financial accounting Financial audit Business entities ( show ) Cooperative Corporation
</item 5>

<item 6>
Board of directors
crisis Sales Security Service Strategic Supply chain Systems administrator Talent Technology Organization ( show ) Architecture Behavior Communication Culture
Conflict Development Engineering Hierarchy Patterns Space Structure Trade ( show ) Business analysis Business ethics Business plan Business judgment rule
Consumer behaviour Business operations International business Business model International trade Business process Business statistics Business and economics
portal Typical duties of boards of directors include : governing the organization by establishing broad policies and setting out strategic objectives ;
selecting , appointing , supporting and reviewing the performance of the chief executive ( of which the titles vary from organization to organization ; the
chief executive may be titled chief executive officer , president or executive director ) ; terminating the chief executive ; ensuring the availability of
adequate financial resources ; approving annual budgets ; accounting to the stakeholders for the organization 's performance ; setting the salaries ,
compensation and benefits of senior management ; The legal responsibilities of boards and board members vary with the nature of the organization , and between
jurisdictions . For companies with publicly trading stock , these responsibilities are typically much more rigorous and complex than for those of other types .
Typically , the board chooses one of its members to be the chairman ( more usually now called the `` chair '' or `` chairperson '' ) , who holds whatever title
is specified in the bylaws or articles of association .
</item 6>

<item 7>
Board of directors
Most organizations have adopted Robert 's Rules of Order as its guide to supplement its own rules . In this book , the rules for conducting board meetings may
be less formal if there is no more than about a dozen board members present . An example of the informality is that motions are not required if it 's clear what
is being discussed . Size ( edit ) Historically , nonprofit boards have not uncommonly had large boards with up to twenty - four members , but a modern trend is
to have smaller boards as small as six or seven people . Studies suggest that after seven people , each additional person reduces the effectiveness of group -
decision - making . Non-corporate boards ( edit ) The role and responsibilities of a board of directors vary depending on the nature and type of business entity
and the laws applying to the entity ( see types of business entity ) . For example , the nature of the business entity may be one that is traded on a public
market ( public company ) , not traded on a public market ( a private , limited or closely held company ) , owned by family members ( a family business ) , or
exempt from income taxes ( a non-profit , not for profit , or tax - exempt entity ) .
</item 7>

<item 8>
Board of directors
According to Investopedia , some analysts think the ideal size is seven . State law may specify a minimum number of directors , maximum number of directors ,
and qualifications for directors ( e.g. whether board members must be individuals or may be business entities ) . Committees ( edit ) While a board may have
several committees , two -- the compensation committee and audit committee -- are critical and must be made up of at least three independent directors and no
inside directors . Other common committees in boards are nominating and governance . Compensation ( edit ) Directors of Fortune 500 companies received median
pay of $234,000 in 2011 . Directorship is a part - time job . A recent National Association of Corporate Directors study found directors averaging just 4.3
hours a week on board work . Surveys indicate that about 20 % of nonprofit foundations pay their board members , and 2 % of American nonprofit organizations do
. 80 % of nonprofit organizations require board members to personally contribute to the organization , as BoardSource recommends . This percentage has increased
in recent years .
</item 8>

<item 9>
Board of directors
There are numerous types of business entities available throughout the world such as a corporation , limited liability company , cooperative , business trust ,
partnership , private limited company , and public limited company . Much of what has been written about boards of directors relates to boards of directors of
business entities actively traded on public markets . More recently , however , material is becoming available for boards of private and closely held businesses
including family businesses . A board - only organization is one whose board is self - appointed , rather than being accountable to a base of members through
elections ; or in which the powers of the membership are extremely limited . Membership organizations ( edit ) In membership organizations , such as a society
made up of members of a certain profession or one advocating a certain cause , a board of directors may have the responsibility of running the organization in
between meetings of the membership , especially if the membership meets infrequently , such as only at an annual general meeting . The amount of powers and
authority delegated to the board depend on the bylaws and rules of the particular organization . Some organizations place matters exclusively in the board 's
control while in others , the general membership retains full power and the board can only make recommendations .
</item 9>

<item 10>
Board of directors
However , in membership organizations , the members elect the president of the organization and the president becomes the chair of the board , unless the bylaws
say otherwise . Directors ( edit ) The directors of an organization are the persons who are members of its board . Several specific terms categorize directors
by the presence or absence of their other relationships to the organization . Inside director ( edit ) An inside director is a director who is also an employee
, officer , chief executive , major shareholder , or someone similarly connected to the organization . Inside directors represent the interests of the entity 's
stakeholders , and often have special knowledge of its inner workings , its financial or market position , and so on .
</item 10>
=== END ITEM id=822a8ef2bdcd52d3 ===
