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

This file has 12 items.

=== ITEM id=40d715be10ccfc67 ===
Gold sentences and facts:
Sentence 0: Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.
  Fact 0.0: Cantarella was a poison
  Fact 0.1: Cantarella was allegedly used by the Borgias
  Fact 0.2: The alleged use of Cantarella by the Borgias occurred during the papacy of Pope Alexander VI
Sentence 1:  The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to remove his name from the records.
  Fact 1.0: The use of Cantarella is not well documented in any of the papal records
  Fact 1.1: Cantarella was most likely conceived after 1503
  Fact 1.2: Cantarella was most likely conceived as part of Pope Julius II's effort to remove his name from the records
Sentence 2: Damnatio memoriae is a modern Latin phrase literally meaning "condemnation of memory", meaning that a person must not be remembered.
  Fact 2.0: Damnatio memoriae is a modern Latin phrase
  Fact 2.1: Damnatio memoriae literally means condemnation of memory
  Fact 2.2: Damnatio memoriae means that a person must not be remembered

Retrieved items:
<item 1>
Dum Diversas
Dum Diversas (English: "Until different") is a papal bull issued on 18 June 1452 by Pope Nicholas V.

 It authorized Afonso V of Portugal to conquer Saracens and pagans and consign them to "perpetual servitude".

 Pope Calixtus III reiterated the bull in 1456 with "Inter Caetera" (not to be confused with Alexander VI's), renewed by Pope Sixtus IV in 1481 and Pope Leo X
in 1514 with "Precelse denotionis".

 The concept of the consignment of exclusive spheres of influence to certain nation states was extended to the Americas in 1493 by Pope Alexander VI with "Inter
caetera".
</item 1>

<item 2>
Felice della Rovere
Felice della Rovere (c. 1483 – September 27, 1536), also known as Madonna Felice, was the illegitimate daughter of Pope Julius II.

 One of the most powerful women of the Italian Renaissance, she was born in Rome around 1483 to Lucrezia Normanni and Cardinal Giuliano della Rovere, later Pope
Julius II.

 Felice was well educated, became accepted into close courtly circles of aristocratic families, and formed friendships with scholars and poets through her
education and genuine interest in humanism.

 Through the influence of her father, including an arranged marriage to Gian Giordano Orsini, she wielded extraordinary wealth and influence both within and
beyond the Roman Curia.

 In particular, she negotiated a peace between Julius II and the Queen of France, and held the position of Orsini Signora for over a decade following the death
of her husband in 1517.

 Felice further increased her power through a castle that she bought with money received from her father, the Castle at Palo, and through her involvement in the
grain trade.
</item 2>

<item 3>
Damnatio memoriae
Damnatio memoriae is a modern Latin phrase literally meaning "condemnation of memory", meaning that a person must not be remembered.
</item 3>

<item 4>
Dictatus papae
Dictatus papae is a compilation of 27 statements of powers arrogated to the Pope that was included in Pope Gregory VII's register under the year 1075.

 Some historians argue that it was written (or dictated) by Gregory VII himself; others argue that it has been inserted in the register at a later date, and
that it had a different origin.

 In 1087 Cardinal Deusdedit published a collection of decretals, dedicated to Pope Victor III, that embodied the law of the Church – Canon law – which he had
compiled from many sources, both legitimate and false (see Pseudo-Isidore).

 The "Dictatus papae" agrees so clearly and closely with this collection that some have argued the "Dictatus" must have been based on it; and so must be of a
later date of compilation and insertion in the papal register than 1087.
</item 4>

<item 5>
Papal conclave, September 1503
The papal conclave of September 1503 elected Pope Pius III to succeed Pope Alexander VI.

 Due to the Italian Wars, the College of Cardinals was surrounded by three potentially hostile armies, loyal to Louis XII of France, Ferdinand II of Aragon, and
Cesare Borgia (the cardinal-nephew of Alexander VI).
</item 5>

<item 6>
Portrait of Pope Julius II
Portrait of Pope Julius II is an oil painting of 1511–12 by the Italian High Renaissance painter Raphael.

 The portrait of Pope Julius II was unusual for its time and would carry a long influence on papal portraiture.

 From early in its life, it was specially hung at the pillars of the church of Santa Maria del Popolo, on the main route from the north into Rome, on feast and
high holy days.

 Giorgio Vasari, writing long after Julius' death, said that "it was so lifelike and true it frightened everyone who saw it, as if it were the living man
himself".
</item 6>

<item 7>
Sicut Dudum
Sicut Dudum (English: "Just As Long Ago" ) is a papal bull promulgated by Pope Eugene IV in Florence on January 13, 1435, which forbade the enslavement of local
natives in the Canary Islands who had converted or were converting to Christianity.

 "Sicut Dudum" was meant to reinforce "Creator Omnium", issued the previous year, condemning Portuguese slave raids in the Canary Islands.

 Over forty years after "Creator Omnium" and "Sicut Dudum, Pope Sixtus IV found it necessary repeat the prohibition in his papal bull "Regimini Gregis" which
threatened the excommunication of all captains or pirates who enslaved Christians.
</item 7>

<item 8>
Cantarella
Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.

 It was probably a variation of arsenic or cantharidin powder (made from blister beetles) .

 The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to
remove his name from the records.
</item 8>

<item 9>
Pedro Luis de Borja Lanzol de Romaní
Pedro Luis de Borja Lanzol de Romaní, O.S.Io.Hieros.

 (1472–1511) was a Roman Catholic cardinal and cardinal-nephew and papal military leader.

 He received a wide variety of sinecures during the papacy of his great-uncle, Pope Alexander VI, but was exiled to Naples on the election of Borja rival Pope
Julius II.

 Borja also fought with the Knights Hospitaller in Jerusalem and Rhodes.
</item 9>

<item 10>
Pope Julius II
Pope Julius II (Italian: "Papa Giulio II" ; Latin: "Iulius II" ) (5 December 1443 – 21 February 1513), born Giuliano della Rovere, and nicknamed "The Fearsome
Pope" and "The Warrior Pope", was Pope from 1 November 1503 to his death in 1513.

 His papacy was marked by an active foreign policy, ambitious building projects, and patronage of the arts—he commissioned the destruction and rebuilding of St.
Peter's Basilica, and Michelangelo's decoration of the ceiling of the Sistine Chapel.

 In addition to an active military policy, he personally led troops into battle on at least two occasions, the first to expel Giovanni Bentivoglio from Bologna
(17 August 1506–23 March 1507), and the second in an attempt to recover Ferrara for the Papal States (1 September 1510–29 June 1512).
</item 10>
=== END ITEM id=40d715be10ccfc67 ===

=== ITEM id=dcb7e02c731ab65e ===
Gold sentences and facts:
Sentence 0: Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.
  Fact 0.0: Francis Keogh Gleason was born on April 14, 1906.
  Fact 0.1: Francis Keogh Gleason died on December 18, 1982.
  Fact 0.2: Francis Keogh Gleason was a resident set decorator at MGM studios.
  Fact 0.3: Francis Keogh Gleason was a resident set decorator at MGM studios for over 40 years.
Sentence 1:  In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi" in 1958) and was nominated an additional 3 times.
  Fact 1.0: Francis Keogh Gleason won 4 Academy Awards during his time at MGM.
  Fact 1.1: Francis Keogh Gleason won an Academy Award for 'An American in Paris' in 1951.
  Fact 1.2: Francis Keogh Gleason won an Academy Award for 'The Bad and the Beautiful' in 1952.
  Fact 1.3: Francis Keogh Gleason won an Academy Award for 'Somebody Up There Likes Me' in 1956.
  Fact 1.4: Francis Keogh Gleason won an Academy Award for 'Gigi' in 1958.
  Fact 1.5: Francis Keogh Gleason was nominated for an Academy Award an additional 3 times.
Sentence 2: Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.
  Fact 2.0: Somebody Up There Likes Me is a 1956 American drama film.
  Fact 2.1: Somebody Up There Likes Me is based on the life of middleweight boxing legend Rocky Graziano.

Retrieved items:
<item 1>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 1>

<item 2>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 2>

<item 3>
Geoffrey Rush
Geoffrey Roy Rush {'1': ", '2': ", '3': ", '4': "} (born 6 July 1951) is an Australian actor and film producer.

 Rush is the youngest amongst the few people who have won the "Triple Crown of Acting": the Academy Award, the Primetime Emmy Award, and the Tony Award.

 He has won one Academy Award for acting (from four nominations), three British Academy Film Awards (from five nominations), two Golden Globe Awards and four
Screen Actors Guild Awards. Rush is the founding President of the Australian Academy of Cinema and Television Arts and was named the 2012 Australian of the
Year.

 He is also the first actor to win the Academy Award, BAFTA Award, Critics' Choice Movie Award, Golden Globe Award, and Screen Actors Guild Award for a single
performance in film for his performance in "Shine" (1996).
</item 3>

<item 4>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 4>

<item 5>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 5>

<item 6>
Somebody Up There Likes Me (1956 film)
Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.

 Joseph Ruttenberg was awarded a 1956 Oscar in the category of Best Cinematography (Black and White).

 The film also won the Oscar for Best Art Direction (Cedric Gibbons, Malcolm F. Brown, Edwin B. Willis, F. Keogh Gleason).

 It was directed by Robert Wise.
</item 6>

<item 7>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 7>

<item 8>
The Man with the Golden Arm
The Man with the Golden Arm is a 1955 American drama film with elements of film noir, based on the novel of the same name by Nelson Algren, which tells the
story of a drug addict who gets clean while in prison, but struggles to stay that way in the outside world.

 Although the addictive drug is never identified in the film, according to the American Film Institute "most contemporary and modern sources assume that it is
heroin", in contrast to Algren's book which named the drug as morphine.

 The film stars Frank Sinatra, Eleanor Parker, Kim Novak, Arnold Stang and Darren McGavin.

 It was adapted for the screen by Walter Newman, Lewis Meltzer and Ben Hecht (uncredited), and directed by Otto Preminger.

 The film's initial release sparked controversy due to its serious, in-depth treatment of the then-taboo subject of drug addiction.
</item 8>

<item 9>
Rocky Graziano
Thomas Rocco Barbella (January 1, 1919 – May 22, 1990), better known as Rocky Graziano, was an American professional boxer who held the World Middleweight
title.

 Graziano is considered one of the greatest knockout artists in boxing history, often displaying the capacity to take his opponent out with a single punch.

 He was ranked 23rd on "The Ring" magazine list of the greatest punchers of all time.

 He fought many of the best middleweights of the era including Sugar Ray Robinson.

 His turbulent and violent life story was the basis of the 1956 Oscar-winning drama film, "Somebody Up There Likes Me", based on his 1955 autobiography of the
same title.
</item 9>

<item 10>
Rocky V
Rocky V is a 1990 American sports drama film.

 It is the fifth film in the "Rocky" series, written by and starring Sylvester Stallone, and co-starring Talia Shire, Stallone's real-life son Sage, and
real-life boxer Tommy Morrison, with Morrison in the role of Tommy Gunn, a talented yet raw boxer.

 Sage played Robert Balboa, whose relationship with his famous father is explored.

 After Stallone directed the second through fourth films in the series, "Rocky V" saw the return of John G. Avildsen, whose direction of "Rocky" won him an
Academy Award for Best Directing.
</item 10>
=== END ITEM id=dcb7e02c731ab65e ===

=== ITEM id=d0aa3d9903aa38a7 ===
Gold sentences and facts:
Sentence 0:  It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center near the Miami River.
  Fact 0.0: It flows in a west to east direction.
  Fact 0.1: It flows from the western part of the state in the Everglades.
  Fact 0.2: It flows past the Miami International Airport.
  Fact 0.3: It flows to a salinity control center near the Miami River.
Sentence 1: The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
  Fact 1.0: The Genesee River is a tributary of Lake Ontario.
  Fact 1.1: The Genesee River flows northward.
  Fact 1.2: The Genesee River flows through the Twin Tiers of Pennsylvania.
  Fact 1.3: The Genesee River flows through the Twin Tiers of New York in the United States.

Retrieved items:
<item 1>
Tamiami Canal
The Tamiami Canal or C-4 Canal, is a canal located in southern Florida in the United States.

 It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center
near the Miami River.

 It averages 8 ft in depth and is over 100 ft wide in some areas.
</item 1>

<item 2>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 2>

<item 3>
Graph query row:
entity: Tamiami Canal
state: Florida
</item 3>

<item 4>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 4>

<item 5>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 5>

<item 6>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 6>

<item 7>
Genesee River
The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
</item 7>

<item 8>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 8>

<item 9>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 9>

<item 10>
Whitewater River (Great Miami River)
The Whitewater River is a 101 mi southerly flowing right tributary of the Great Miami River in southeastern Indiana and southwestern Ohio in the United States.

 It is formed by the confluence of two forks, the West Fork and East Fork.

 The name is a misnomer, as there is no true white water on the river.

 However, there are many rapids due to the steep gradient present - the river falls an average of 6 ft/mile .

 The gradient rendered upstream navigation impossible, and in the mid-nineteenth century resulted in the construction of the Whitewater Canal paralleling the
river from north of Connersville, Indiana, to the Ohio River.
</item 10>
=== END ITEM id=d0aa3d9903aa38a7 ===

=== ITEM id=0516c698efeac08a ===
Gold sentences and facts:
Sentence 0:  He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink, T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
  Fact 0.0: This producer has produced songs for Justin Bieber.
  Fact 0.1: This producer has produced songs for Monica.
  Fact 0.2: This producer has produced songs for Britney Spears.
  Fact 0.3: This producer has produced songs for Usher.
  Fact 0.4: This producer has produced songs for Keri Hilson.
  Fact 0.5: This producer has produced songs for T.I.
  Fact 0.6: This producer has produced songs for Nelly Furtado.
  Fact 0.7: This producer has produced songs for Kevin Cossom.
  Fact 0.8: This producer has produced songs for Ciara.
  Fact 0.9: This producer has produced songs for Mariah Carey.
  Fact 0.10: This producer has produced songs for Timbaland.
  Fact 0.11: This producer has produced songs for Madonna.
  Fact 0.12: This producer has produced songs for J. Cole.
  Fact 0.13: This producer has produced songs for Kanye West.
  Fact 0.14: This producer has produced songs for Whitney Houston.
  Fact 0.15: This producer has produced songs for Missy Elliott.
  Fact 0.16: This producer has produced songs for M.I.A.
  Fact 0.17: This producer has produced songs for Justin Timberlake.
  Fact 0.18: This producer has produced songs for JoJo.
  Fact 0.19: This producer has produced songs for Joe Jonas.
  Fact 0.20: This producer has produced songs for Simple Plan.
  Fact 0.21: This producer has produced songs for The Clutch.
  Fact 0.22: This producer has produced songs for Pink.
  Fact 0.23: This producer has produced songs for T-Pain.
  Fact 0.24: This producer has produced songs for Diddy.
  Fact 0.25: This producer has produced songs for Meek Mill.
  Fact 0.26: This producer has produced songs for Bjork.
  Fact 0.27: This producer has produced songs for Duran Duran.
Sentence 1:  Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.
  Fact 1.0: Cole was raised in Fayetteville, North Carolina.
  Fact 1.1: Cole initially gained recognition as a rapper.
  Fact 1.2: Cole initially gained recognition as a rapper after the release of his debut mixtape.
  Fact 1.3: Cole's debut mixtape is titled The Come Up.
  Fact 1.4: Cole's debut mixtape The Come Up was released in early 2007.

Retrieved items:
<item 1>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 1>

<item 2>
Anna Graceman
Anna Graceman is a young American singer-songwriter and pianist whose self-penned songs and music have gained attention at home and abroad.

 Her style of music incorporates elements of folk, soul and rock.

 As of 2016 she had written and performed over 70 of her own published songs; writing as well as producing her releases.

 Anna continues to divide her time between her solo projects and her new project - the ‘Graceman’ Band - a trio that includes Allie Graceman (Rhythm/Lead
Guitar) and Landon Graceman (Drums).
</item 2>

<item 3>
Daniel Boone Native Gardens
Daniel Boone Native Gardens, located in Boone, North Carolina, United States, has a collection of North Carolina native plants in an informal landscaped design.

 The gardens are open daily from May to October.
</item 3>

<item 4>
Lil Ru
Sylvester Samuels better known by his stage name Lil Ru, is an American rapper from Ridgeway, South Carolina currently signed to Def Jam Recordings.

 His debut album, 21 & Up was released on August 25, 2009.

 The Ridgeway native was 16 when he made his professional foray into the music business.

 Inspired by New Orleans’ innovative Cash Money Crew, Ru began making a name for himself on his local music scene, doing live shows and pressing up his own CDs.

 His hard-hitting lyrics and entrepreneurial spirit caught the attention of fellow South Carolinian Angie Stone.

 Shortly after the neo-soul songstress helped him secure a deal with Elektra Records, Ru found himself unsigned again, among the artists lost in the shuffle
after the label merged with Atlantic Records.

 Music fans first heard him on his 2001 debut single Will Destroy.

 He then released his 2002 follow up, Shawty What You Doin’.

 Both songs reached the Billboard R&B/Hip-Hop charts and helped land him at his next label home, Capitol Records.
</item 4>

<item 5>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 5>

<item 6>
The Foundation (2008), Zac Brown Band’s major-label debut, sold 3.4 million US copies by 2015, earned Grammy and ACM nominations, and won the band a Best New
Artist Grammy. Tevin Campbell’s Babyface-produced “Can We Talk” reached Hot 100 #9, topped the R&B chart for three weeks, went gold, and won a Soul Train award.
John Cena’s 2005 album introduced his entrance theme; Bow Wow’s “Fresh Azim
</item 6>

<item 7>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 7>

<item 8>
Profiles span hip-hop, pop, rock and electronic music. Method Man is a Wu-Tang Clan rapper and Grammy winner; Mick is a celebrity DJ and entrepreneur; eLDee is
a former Nigerian rapper turned IT consultant; Lil Ru is a South Carolina rapper; J. Cole is a rapper-producer. Others include producers Illmind, Danja, Zach
Crowell and Raleigh Moncrief; performers Luther Campbell, Dappy, Zhu and Daron Jones; and DJ Lance Rock. Albums
</item 8>

<item 9>
Raleigh Moncrief
Raleigh Moncrief is an American electronic musician, record producer and recording engineer who has worked with Zach Hill, Dirty Projectors and !!!

.
</item 9>

<item 10>
Kaydean
Kaydean is a British-American record and TV producer, composer, DJ, record label executive and artist manager.

 Kaydean produces, composes and arranges music in the Dance, Electronica, Latin Pop, Contemporary R&B, and Electronica genres.

 Born in London, he has worked with many artists including Janet Jackson, Maxi Priest, Robyn, Man Parrish, Angel "Cucco" Peña, Baron Lopez, Raekwon, Ky-Mani
Marley, Frankie Cutlass and Nocera amongst others.

 He is also known for discovering Latin Pop Star Janid and managing her career.

 He produced her Spanish album "La Magia" under the Sony Music Latin imprint label, Handshake and her Hit Single "Penicilina".

 In 2015 he debuted as a TV executive producer with Janid's reality show on Mega TV, The first episode aired on September 6, 2015.
</item 10>
=== END ITEM id=0516c698efeac08a ===

=== ITEM id=128505da3817ac82 ===
Gold sentences and facts:
Sentence 0: Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France (1933–1940 and 1950–1957), and the United States (1947–1950).
  Fact 0.0: Maximillian Oppenheimer was born on 6 May 1902.
  Fact 0.1: Maximillian Oppenheimer died on 26 March 1957.
  Fact 0.2: Maximillian Oppenheimer was known as Max Ophüls.
  Fact 0.3: Max Ophüls was a German-born film director.
  Fact 0.4: Max Ophüls worked in Germany from 1931 to 1933.
  Fact 0.5: Max Ophüls worked in France from 1933 to 1940.
  Fact 0.6: Max Ophüls worked in France from 1950 to 1957.
  Fact 0.7: Max Ophüls worked in the United States from 1947 to 1950.
Sentence 1: Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.
  Fact 1.0: Shirley Clarke was born on October 2, 1919.
  Fact 1.1: Shirley Clarke died on September 23, 1997.
  Fact 1.2: Shirley Clarke was an American experimental filmmaker.
  Fact 1.3: Shirley Clarke was an American independent filmmaker.

Retrieved items:
<item 1>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 1>

<item 2>
Lois Weber
Lois Weber (June 13, 1879 – November 13, 1939) was an American silent film actress, screenwriter, producer, and director, who is considered "the most important
female director the American film industry has known", and "one of the most important and prolific film directors in the era of silent films".

 Film historian Anthony Slide asserts that: "Along with D.W. Griffith, Weber was the American cinema's first genuine auteur, a filmmaker involved in all aspects
of production and one who utilized the motion picture to put across her own ideas and philosophies."
</item 2>

<item 3>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 3>

<item 4>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 4>

<item 5>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 5>

<item 6>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 6>

<item 7>
Alain Jessua
Alain Jessua (born 16 January 1932) is a French film director and screenwriter.

 He directed ten films between 1956 and 1997.

 He worked as assistant director for Jacques Becker on the set of "Casque d'or", with Max Ophüls for "Madame de..." and "Lola Montès" and with Marcel Carné on
"Wasteland". "

Léon la lune" his first short film won the influential Prix Jean Vigo in 1957.

 He directed first feature film in 1963 "La vie à l'envers" that won Best First Film at Venice Film Festival, in 1964.
</item 7>

<item 8>
Max Ophüls
Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France
(1933–1940 and 1950–1957), and the United States (1947–1950).

 He made nearly 30 films, the latter ones being especially notable: "La Ronde" (1950), "Le Plaisir" (1952), "The Earrings of Madame de..." (1953) and "Lola
Montès" (1955).
</item 8>

<item 9>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 9>

<item 10>
The Earrings of Madame de…
The Earrings of Madame de… (French: Madame de… ] ) is a 1953 drama film directed by Max Ophüls, adapted from Louise Leveque de Vilmorin's period novel by
Ophüls, Marcel Archard and Annette Wadement.

 The film is considered a masterpiece of the 1950s French cinema.

 Andrew Sarris called it "the most perfect film ever made".

 Ophüls said the story's construction attracted him, stating "there is always the same axis around which the action continually turns like a carousel.

 A tiny, scarcely visible axis: a pair of earrings."
</item 10>
=== END ITEM id=128505da3817ac82 ===

=== ITEM id=7a5a52aed476d1b8 ===
Gold sentences and facts:
Sentence 0: Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of the indie rock band Bloc Party.
  Fact 0.0: Kelechukwu 'Kele' Okereke was born on 13 October 1981.
  Fact 0.1: Kelechukwu 'Kele' Okereke is also known by the mononym Kele.
  Fact 0.2: Kelechukwu 'Kele' Okereke is a British musician.
  Fact 0.3: Kelechukwu 'Kele' Okereke is best known as the lead singer and rhythm guitarist of the indie rock band Bloc Party.
Sentence 1: Jon Hume (born 6 November 1983) is the lead singer of Evermore.
  Fact 1.0: Jon Hume was born on 6 November 1983.
  Fact 1.1: Jon Hume is the lead singer of Evermore.
Sentence 2:  Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
  Fact 2.0: Jon is the eldest of the three brothers.
  Fact 2.1: The three brothers make up the New Zealand band Evermore.
  Fact 2.2: Evermore is a New Zealand band.

Retrieved items:
<item 1>
Graph query row:
entity1: Jon Hume
entity2: Evermore
relationships: ['MEMBER_OF']
connected_entities: ['Jon Hume', 'Evermore']
</item 1>

<item 2>
Evermore (Evermore album)
Evermore is the self-titled international debut album by New Zealand rock band Evermore, which was also released in Australia and New Zealand on 12 March 2010.

 The album is a collection of songs from the band's entire history, as well as three new songs including the single "Underground".
</item 2>

<item 3>
Dann Hume
Dann Hume (born Daniel Benjamin Cobbe, 1 September 1987, Whangaparaoa, New Zealand) is a singer-songwriter and record producer.

 Hume is the youngest of three brothers (with Peter Hume and Jon Hume) who make up the alternative rock band, Evermore since 1999.
</item 3>

<item 4>
Between the Lines (Evermore song)
"Between the Lines" is the first single by Evermore, taken from their third studio album "".

 Evermore's Jon Hume said that "Between the Lines was the first song to come out of a search for a new musical experience as a band."

 It was released as a free download on Evermore's official website on 10 November 2008.
</item 4>

<item 5>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 5>

<item 6>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 6>

<item 7>
Light Surrounding You
"Light Surrounding You" is the second single by alternative rock band, Evermore, taken from their second studio album, "Real Life" (June 2006).

 It was written by Dann Hume, the group's drummer and lead singer and guitarists Jon Hume.

 The record is co-produced by Jon Hume.

 The song was released in October 2006 and peaked at No. 15 on the RIANZ Singles Chart in New Zealand and number one on the ARIA Singles Chart, making it
Evermore's most successful single in Australia.

 It was the first single by a New Zealand artist to top the Australian charts since "How Bizarre" by OMC in 1996.
</item 7>

<item 8>
Jon Hume
Jon Hume (born 6 November 1983) is the lead singer of Evermore.

 He is a singer-songwriter, multi-instrumentalist, Record Producer and Studio owner.

 Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
</item 8>

<item 9>
Kele Okereke
Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of
the indie rock band Bloc Party.
</item 9>

<item 10>
Running (Evermore song)
"Running" is the lead single by alternative rock band Evermore, taken from their second studio album, "Real Life" (July 2006).

 It was written by backing vocalist and drummer, Dann Hume.

 It was issued on 3 June, ahead of the album.

 It was co-produced by Jon with John Alagía (Dave Matthews Band, John Mayer).

 It peaked at No. 4 on their native RIANZ Singles Chart in New Zealand and No. 5 on their adopted country of Australia's ARIA Singles Chart.
</item 10>
=== END ITEM id=7a5a52aed476d1b8 ===

=== ITEM id=27a19546f9dd3827 ===
Gold sentences and facts:
Sentence 0:  Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
  Fact 0.0: Dan Hurley is the son of Bob Hurley.
  Fact 0.1: Bob Hurley is a Hall of Fame high school coach.
  Fact 0.2: Dan Hurley is the younger brother of Bobby Hurley.
  Fact 0.3: Bobby Hurley is a former Duke guard.
  Fact 0.4: Bobby Hurley is a former Sacramento Kings guard.
Sentence 1: Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.
  Fact 1.0: Robert Matthew Hurley was born on June 28, 1971.
  Fact 1.1: Robert Matthew Hurley is an American basketball coach.
  Fact 1.2: Robert Matthew Hurley is a former college player.
  Fact 1.3: Robert Matthew Hurley is a former professional player.
Sentence 2:  Hurley is currently the head coach of the Arizona State men's team.
  Fact 2.0: Hurley is currently the head coach of the Arizona State men's team.

Retrieved items:
<item 1>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 1>

<item 2>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 2>

<item 3>
Bob Hurley
Robert Emmet Hurley, Sr. (born July 31, 1947) is a basketball coach at St. Anthony High School in Jersey City, New Jersey.

 Hurley has amassed 28 state championships and more than 1000 wins in 39 years as a coach.

 On February 2, 2011, Hurley became the tenth coach in high school history to win 1000 games.

 He is featured in the documentary "The Street Stops Here".

 Five of his teams have gone undefeated.

 On April 5, 2010, he was announced as the only coach to be inducted into the Naismith Memorial Basketball Hall of Fame that year and only the third high school
coach in history to be so honored; he was formally inducted on August 13 of that year.

 Hurley is the father of Bobby Hurley, a former All-American point guard at Duke and the head basketball coach at Arizona State, and Dan Hurley, who was hired
in February 2012 to coach the University of Rhode Island after two years of coaching at Wagner College and nine years coaching at Newark's Saint Benedict's
Preparatory School, also one of the top high school programs in the nation.
</item 3>

<item 4>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 4>

<item 5>
Rhode Island Rams men's basketball
The Rhode Island Rams men's basketball team is a college basketball program that competes in NCAA Division I and the Atlantic 10 Conference representing the
University of Rhode Island.

 The team is under the direction of head coach Dan Hurley, who was hired on March 20, 2012.

 The Rams play their home games at the Ryan Center (capacity 7,657).
</item 5>

<item 6>
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 6>

<item 7>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 7>

<item 8>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 8>

<item 9>
2015–16 Iowa State Cyclones men's basketball team
The 2015–16 Iowa State Cyclones men's basketball team represents Iowa State University during the 2015–16 NCAA Division I men's basketball season.

 The Cyclones were coached by Steve Prohm, who was in his 1st season.

 They played their home games at Hilton Coliseum in Ames, Iowa and competed in the Big 12 Conference.
</item 9>

<item 10>
Nate Oats
Nathanael "Nate" Oats (born October 13, 1974) is an American basketball coach.

 He is currently the head basketball coach at the State University of New York at Buffalo, following two seasons as an assistant coach under Bobby Hurley.

 Oats was named head coach on April 11, 2015 after Hurley was hired by Arizona State as head coach.

 In only his first season as head coach, he took the Buffalo Bulls to their second straight NCAA Tournament bid after winning the Mid-American Conference
Championship.
</item 10>
=== END ITEM id=27a19546f9dd3827 ===

=== ITEM id=3f5e21cb782e48a5 ===
Gold sentences and facts:
Sentence 0: San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San Antonio metropolitan area.
  Fact 0.0: San Antonio International Airport has the IATA code SAT.
  Fact 0.1: San Antonio International Airport has the ICAO code KSAT.
  Fact 0.2: San Antonio International Airport has the FAA LID SAT.
  Fact 0.3: San Antonio International Airport is an international airport.
  Fact 0.4: San Antonio International Airport is located in San Antonio, Texas.
  Fact 0.5: San Antonio International Airport serves the Greater San Antonio metropolitan area.
Sentence 1: Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.
  Fact 1.0: Yakutat Airport has the IATA code YAK.
  Fact 1.1: Yakutat Airport has the ICAO code PAYA.
  Fact 1.2: Yakutat Airport has the FAA LID YAK.
  Fact 1.3: Yakutat Airport is a state owned, public use airport.
  Fact 1.4: Yakutat Airport is located three nautical miles (6 km) southeast of the central business district of Yakutat.
  Fact 1.5: Yakutat is a city and borough in the U.S. state of Alaska.
  Fact 1.6: Yakutat has no road access to the outside world.

Retrieved items:
<item 1>
Yakutat Airport
Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business
district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.

 Airline service is subsidized by the Essential Air Service program.
</item 1>

<item 2>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 2>

<item 3>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 3>

<item 4>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 4>

<item 5>
Ta'if Regional Airport
Ta'if Regional Airport (IATA: TIF, ICAO: OETF) is an airport in Ta'if, Saudi Arabia.

 Despite its name, it also offers many international flights as well.

 The airport is located 30 km to the east of Taif and 70 km from Mecca.

 The airport is considered important in Saudi aviation history as it witnessed the first landing of Ibn Saud's plane, founder of the Kingdom.

 It was converted to a regional airport in 2009 when GACA allowed international airlines to operate at the airport as the city's population was increasing and
to reduce pressure on the three main airports at the time.

 Despite being named as a "Regional" airport, the airport actually has international flights destinations in half a dozen countries outside of Saudi Arabia.
</item 5>

<item 6>
Pago Pago International Airport
Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of
the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the
United States.
</item 6>

<item 7>
Graph query row:
airport: San Antonio International Airport
location: Texas
country: United States
</item 7>

<item 8>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 8>

<item 9>
San Antonio International
San Antonio International was an American soccer club based in San Antonio, Texas that was a member of the Lone Star Soccer Alliance.
</item 9>

<item 10>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 10>
=== END ITEM id=3f5e21cb782e48a5 ===

=== ITEM id=c9a6d2732fae9633 ===
Gold sentences and facts:
Sentence 0: The Apple Dumpling Gang is a 1975 American comedy-western film produced by Walt Disney Productions about a slick gambler named Russell Donovan (Bill Bixby) who is duped into taking care of a group of orphans who eventually strike gold during the California Gold Rush.
  Fact 0.0: The Apple Dumpling Gang is a 1975 American comedy-western film.
  Fact 0.1: The Apple Dumpling Gang was produced by Walt Disney Productions.
  Fact 0.2: The Apple Dumpling Gang is about a slick gambler named Russell Donovan.
  Fact 0.3: Russell Donovan is played by Bill Bixby.
  Fact 0.4: Russell Donovan is duped into taking care of a group of orphans.
  Fact 0.5: The group of orphans eventually strikes gold during the California Gold Rush.
Sentence 1: Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.
  Fact 1.0: Heavyweights is a 1995 American comedy-drama film.
  Fact 1.1: Heavyweights was directed by Steven Brill.
  Fact 1.2: Heavyweights was written by Steven Brill with Judd Apatow.

Retrieved items:
<item 1>
Heavyweights
Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.

 The film centers around a fat camp for kids that is taken over by a fitness guru named Tony Perkis (Ben Stiller).
</item 1>

<item 2>
Graph query row:
film: Heavyweights
release_year: 1995
</item 2>

<item 3>
Buddy Baker (composer)
Norman Dale "Buddy" Baker (January 4, 1918 – July 26, 2002) was an American composer who, together with Paul J. Smith, scored many Disney films, such as "The
Apple Dumpling Gang" in 1975, "The Apple Dumpling Gang Rides Again" in 1979, "The Shaggy D.A." in 1976, "The Many Adventures of Winnie the Pooh" in 1977, and
"The Fox and the Hound" in 1981.
</item 3>

<item 4>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 4>

<item 5>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 5>

<item 6>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 6>

<item 7>
The Apple Dumpling Gang
The Apple Dumpling Gang is a 1971 novel by Jack Bickham, about a group of orphaned children during the California gold rush.

 They encounter a gambler who reluctantly helps them, as well as a pair of hapless robbers who are after the gold the children have found.
</item 7>

<item 8>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 8>

<item 9>
The Apple Dumpling Gang (film)
The Apple Dumpling Gang is a 1975 American comedy-western film produced by Walt Disney Productions about a slick gambler named Russell Donovan (Bill Bixby) who
is duped into taking care of a group of orphans who eventually strike gold during the California Gold Rush.
</item 9>

<item 10>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 10>
=== END ITEM id=c9a6d2732fae9633 ===

=== ITEM id=410e45bef8d0e894 ===
Gold sentences and facts:
Sentence 0: Catcha Lake is a Canadian lake located in the central part of Nova Scotia's Halifax Regional Municipality.
  Fact 0.0: Catcha Lake is a Canadian lake
  Fact 0.1: Catcha Lake is located in the central part of Nova Scotia's Halifax Regional Municipality
Sentence 1: The Hedley Mascot Mine was a gold mine in Hedley, British Columbia, Canada.
  Fact 1.0: The Hedley Mascot Mine was a gold mine
  Fact 1.1: The Hedley Mascot Mine was in Hedley, British Columbia, Canada

Retrieved items:
<item 1>
Catcha Lake
Catcha Lake is a Canadian lake located in the central part of Nova Scotia's Halifax Regional Municipality.
</item 1>

<item 2>
Cassiope lycopodioides
Cassiope lycopodioides, Haida Gwaii mountain-heather or clubmoss mountain heather, is a plant species native to southern Alaska, British Columbia, and the US
State of Washington.

 It is found on rocky slopes in arctic and alpine tundra at elevations up to 2000 m.

 In Washington, it is reported only from King County.

 The specific epithet ""lycopodioides"" refers to the plant's superficial resemblance to some species of clubmoss "(Lycopodium" sensu lato).
</item 2>

<item 3>
Beaconsfield, Victoria
Beaconsfield is a satellite suburb of Melbourne, Victoria, Australia, 46 km south-east of Melbourne's Central Business District, located within the Shire of
Cardinia local government area.

 Beaconsfield recorded a population of 6,714 at the 2016 Census.
</item 3>

<item 4>
Hounsfield Heights/Briar Hill, Calgary
Hounsfield Heights/Briar Hill is an inner suburban neighbourhood in northwest Calgary, Alberta, Canada.

 Located north of the Hillhurst and West Hillhurst communities, the boundaries of the district are 16th Avenue N (Trans-Canada Highway)to the north; 14th Street
W to the east; Lane north of 7th Avenue N to 19th Street W and 8th Avenue N to the south; and Crowchild Trail, 12th Avenue N, Juniper Road, and 22nd Street W to
the west.

 Lions Park C-Train station is located within the community.

 The community is built on an escarpment and is popular for its views of downtown to the south and the Rocky Mountains to the west.
</item 4>

<item 5>
Victor Diamond Mine
The Victor Diamond Mine is the first Canadian diamond mine located in Ontario, and De Beers' second diamond mine in Canada (after the Snap Lake Diamond Mine).

 It is located in the Northern Ontario Ring of Fire, in the James Bay Lowlands 90 km west of Attawapiskat in the remote northern part of the province.

 In June 2005, the Attawapiskat First Nation voted in favour (85.5%) of ratifying the Impact Benefit Agreement (IBA).

 Construction of the mine began in February 2006 which created 3200 positions; mining and operations will create around 400 permanent positions.

 The Victor Mine is an open-pit mine, with a processing plant, workshops, and an airstrip located on site.

 By 2013-2014 royalties collected from De Beers Victor Diamond Mine amounted to $226.

 At that time De Beers was continuing to pay off its "$1 billion investment to build the mine and from now until it closes, the company expects to pay tens of
millions of dollars in royalties."
</item 5>

<item 6>
Lake Shore Mine
The Lake Shore Mine is a gold mine located in Kirkland Lake, Ontario.

 In July 1912, Harry Oakes staked claims L-2605-6 which were in the lake itself and had reverted for non-performance of work.

 On September 6, 1912, he registered the transfer of claim L-1557 that Melville McDougall had staked for Oakes previously.

 On September 23, 1911, Harry Oakes registered the transfer of claim T-16635 from George Minaker and named the property Lake Shore.

 In production from 1918 to 1965, the mine produced almost 8.5 million ounces of gold and represented over a third of the gold produced in the entire camp.
</item 6>

<item 7>
Leviathan Mine
Leviathan Mine is a United States superfund site (CERCLIS ID: CAD98067685) at an abandoned open-pit sulfur mine located in Alpine County, California.

 The mine is located at on the eastern slope of the Sierra Nevada at about 7,000 ft elevation, 6 mi east of Markleeville and 24 mi southeast of Lake Tahoe.

 The mine site comprises approximately 250 acre of land surrounded by the Toiyabe National Forest, which is only accessible a few months a year.

 The approximately 22 million tons of sulfur ore-containing crushed rock at the mine are responsible for contaminating the Leviathan and Aspen Creek, which join
with Mountaineer Creek to form Bryant Creek which ultimately empties into the East Fork of the Carson River.

 These water bodies are listed as 303(d) impaired.

 The site location is seismically active.
</item 7>

<item 8>
Hedley Mascot Mine
The Hedley Mascot Mine was a gold mine in Hedley, British Columbia, Canada.

 Gold was first discovered in the Nickel Plate Mountain area in 1897 and several small mines were developed over the years.

 The Hedley Mascot Mine operated between 1936 and 1949 and was one of the most unusual mining operations in the world, being built entirely on the side of a
mountain, 5,000 feet above the town of Hedley or seven thousand feet above sea level.

 In the 1990s, the British Columbia government was going to burn the site down because it posed a safety risk, but the Minister of Tourism at the time
intervened and, in 1995, steps were taken to preserve the site as a Provincial Heritage resource.
</item 8>

<item 9>
Hedley, British Columbia
Hedley is an unincorporated town in southern British Columbia, Canada, named after Robert R. Hedley, the manager of the Hall Smelter in Nelson.

 Hedley is located at the foot of Nickel Plate Mountain in the Similkameen.

 The town had a population of approximately 400 as of 2005.

 In the early 1900s, Hedley's population peaked over 1,000 people, primarily due to the gold mining industry.

 The Hedley Museum and the Mascot Mine Museum display artifacts and photographs from this era.
</item 9>

<item 10>
Electoral district of Bass
The Electoral district of Bass is an electoral district of the Victorian Legislative Assembly.

 It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip
Island.

 It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi,
and part of suburban Pakenham.

 It lies within the Eastern Victoria Region of the upper house, the Legislative Council.
</item 10>
=== END ITEM id=410e45bef8d0e894 ===

=== ITEM id=91db521764c98896 ===
Gold sentences and facts:
Sentence 0: Park Sung-woong (born January 9, 1973) is a South Korean actor.
  Fact 0.0: Park Sung-woong was born on January 9, 1973.
  Fact 0.1: Park Sung-woong is a South Korean actor.
Sentence 1:  Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a serial killer in "The Deal" (2015).
  Fact 1.0: Park Sung-woong made his acting debut in 'No. 3' in 1997.
  Fact 1.1: Park Sung-woong has starred in several movies and television series.
  Fact 1.2: Park Sung-woong notably starred as a gangster in 'New World' (2013).
  Fact 1.3: Park Sung-woong notably starred as a serial killer in 'The Deal' (2015).
Sentence 2: The Deal (; lit.
  Fact 2.0: The Deal's literal translation is given in the next sentence, after the 'lit.' marker.
Sentence 3:  "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
  Fact 3.0: The literal translation of The Deal is 'Murder Request'.
  Fact 3.1: The Deal is a 2015 South Korean crime thriller film.
  Fact 3.2: The Deal was directed by Son Yong-ho.

Retrieved items:
<item 1>
Hidden Identity (TV series)
Hidden Identity () is a 2015 South Korean television series starring Kim Bum, Park Sung-woong, Yoon So-yi and Lee Won-jong.

 It aired on tvN from June 16 to August 4, 2015 on Mondays and Tuesdays at 23:00 for 16 episodes.
</item 1>

<item 2>
A Violent Prosecutor
A Violent Prosecutor is a 2016 South Korean crime film directed by Lee Il-hyung, produced by Guk Su Ran and starring Hwang Jung-min, Kang Dong-won, Lee Sung-min
and Park Sung-woong.

 It was released in South Korea on February 3, 2016 by Showbox.
</item 2>

<item 3>
The Swindlers (2017 film)
The Swindlers is an upcoming South Korean crime drama film directed by Jang Chang-won.

 The film stars Hyun Bin, Yoo Ji-tae, Bae Seong-woo, Park Sung-woong, Nana and Ahn Se-ha.
</item 3>

<item 4>
A Company Man
A Company Man () is a 2012 South Korean action thriller film starring So Ji-sub, Lee Mi-yeon, Kwak Do-won and Kim Dong-jun.

 It is about a hitman who find himself targeted by his ex-employers after he falls in love with a single mother and quits his job.
</item 4>

<item 5>
Deep Trap
Deep Trap () (previously known as Exchange) is a 2015 South Korean crime thriller film starring Ma Dong-seok, Jo Han-sun and Kim Min-kyung, and directed by Kwon
Hyung-jin.

 Based on an actual SNS crime, it depicts the terror experienced by a married couple while on a trip to an isolated island.

 It won Best Film Award in the Orient Express section at the Fantasporto in 2016.
</item 5>

<item 6>
The Boys Who Cried Wolf
The Boys Who Cried Wolf (), also known as The Shepherd, is a 2015 South Korean crime thriller drama film.

 Written and directed by Kim Jin-hwang in his first feature-length for his Korean Academy of Film Arts (KAFA)'s final year undergraduate film project, it
depicts how a former stage actor becomes involved in a murder case.
</item 6>

<item 7>
The Deal (2015 film)
The Deal (; lit.

 "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
</item 7>

<item 8>
Park Sung-woong
Park Sung-woong (born January 9, 1973) is a South Korean actor.

 Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a
serial killer in "The Deal" (2015).
</item 8>

<item 9>
Office (2015 South Korean film)
Office (오피스 ) is 2015 South Korean slasher-thriller film directed by Hong Won-chan and starring Go Ah-sung and Park Sung-woong.

 It is about a detective trying to figure out why a mild-mannered man has killed his family and is targeting his co-workers.

 The film premiered at the Midnight Screenings section of the 2015 Cannes Film Festival.
</item 9>

<item 10>
Fatal Intuition
Fatal Intuition (; lit.

 "It's Him") is a 2015 South Korean crime thriller film written and directed by Yun Jun-hyeong.
</item 10>
=== END ITEM id=91db521764c98896 ===

=== ITEM id=35c3badef49e1ab4 ===
Gold sentences and facts:
Sentence 0: James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more than 70 productions to his credit.
  Fact 0.0: James David Sharman was born on 12 March 1945.
  Fact 0.1: James David Sharman is the son of boxing tent entrepreneur Jimmy Sharman.
  Fact 0.2: James David Sharman is an Australian director.
  Fact 0.3: James David Sharman is an Australian writer for film and stage.
  Fact 0.4: James David Sharman has more than 70 productions to his credit.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born on August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is an American director.
  Fact 1.2: Ava Marie DuVernay is an American screenwriter.
  Fact 1.3: Ava Marie DuVernay is an American film marketer.
  Fact 1.4: Ava Marie DuVernay is an American film distributor.

Retrieved items:
<item 1>
Graph query row:
person: James David Sharman
occupation: ['director and writer for film and stage', 'theatre director']
</item 1>

<item 2>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 2>

<item 3>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 3>

<item 4>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 4>

<item 5>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 5>

<item 6>
♯JusticeForFlint
Initiated by "Creed" director Ryan Coogler and "Selma" director Ava DuVernay, the free public event was held at the Whiting Auditorium in Flint, Michigan.

 Hosted by comedian Hannibal Buress, it featured singers Janelle Monáe and Ledisi, as well as actor-activists Jesse Williams and Jussie Smollett, amongst
others.

 Surprising star guest was "King of Motown" Stevie Wonder, who teamed up with Monáe for "Hell You Talkin' Bout", a protest song she had written for the people
of Flint.
</item 6>

<item 7>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 7>

<item 8>
Jim Sharman
James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more
than 70 productions to his credit.

 He is renowned in Australia for his work as a theatre director from the 1960s to the present, and is best known internationally as the director of the 1973
theatrical hit "The Rocky Horror Show", its film adaptation "The Rocky Horror Picture Show" (1975) and the film's follow-up "Shock Treatment" (1981).
</item 8>

<item 9>
Graph query row:
person: Ava DuVernay
occupation: ['director', 'screenwriter', 'film marketer', 'film distributor']
</item 9>

<item 10>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 10>
=== END ITEM id=35c3badef49e1ab4 ===
