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

This file has 11 items.

=== ITEM id=003a3011b5f1695f ===
Gold sentences and facts:
Sentence 0: The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.
  Fact 0.0: The Bears and I is a 1974 American drama film.
  Fact 0.1: The Bears and I was directed by Bernard McEveety.
  Fact 0.2: The Bears and I was written by John Whedon.
Sentence 1:  The film was released on July 31, 1974, by Buena Vista Distribution.
  Fact 1.0: The Bears and I was released on July 31, 1974.
  Fact 1.1: The Bears and I was released by Buena Vista Distribution.
Sentence 2: Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as co-director.
  Fact 2.0: Oceans (French: Océans) is a 2009 French nature documentary film.
  Fact 2.1: Oceans was directed by Jacques Perrin.
  Fact 2.2: Oceans was produced by Jacques Perrin.
  Fact 2.3: Oceans was co-written by Jacques Perrin.
  Fact 2.4: Oceans was narrated by Jacques Perrin.
  Fact 2.5: Jacques Cluzaud was co-director of Oceans.

Retrieved items:
<item 1>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 1>

<item 2>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 2>

<item 3>
Return to the Blue Lagoon
Return to the Blue Lagoon is a 1991 American romance and adventure film starring Milla Jovovich and Brian Krause, produced and directed by William A. Graham.

 The film is a sequel to "The Blue Lagoon" (1980).

 The screenplay by Leslie Stevens was based on the novel "The Garden of God" by Henry De Vere Stacpoole.

 The original music score was composed by Basil Poledouris.

 The film's closing theme song "A World of Our Own" is performed by Surface featuring Bernard Jackson.

 The music was written by Barry Mann, and the lyrics were written by Cynthia Weil.
</item 3>

<item 4>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 4>

<item 5>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 5>

<item 6>
Entre el Cielo y el Infierno
Entre el Cielo y el Infierno (Between Heaven and Hell) is the fifth studio album by Argentine heavy metal band Rata Blanca.

 The album was recorded at Estudios La Nave between June and July ,1994 and was released in September, 1994 by BMG.

 Is the first and unique album featuring vocalist Mario Ian and also is the first album featuring Javier Retamozo on keyboards.
</item 6>

<item 7>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 7>

<item 8>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 8>

<item 9>
I Against I
I Against I is the third studio album by the American hardcore punk band Bad Brains.

 It was released in November 1986 through SST Records with the catalog number SST 065.

 The best-selling album in the band's catalog, "I Against I" is a critically acclaimed album that mixes American hardcore punk with funk, soul, reggae and heavy
metal.

 It is also included in the book "1001 Albums You Must Hear Before You Die".

 The album featured an MTV video for the title track directed by Paul Rachman who later went on to direct the documentary feature film "American Hardcore".
</item 9>

<item 10>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 10>
=== END ITEM id=003a3011b5f1695f ===

=== ITEM id=bb27521b42db68b1 ===
Gold sentences and facts:
Sentence 0: Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.
  Fact 0.0: Pinhead Gunpowder is an American punk rock band.
  Fact 0.1: Pinhead Gunpowder formed in East Bay, California, in 1990.
Sentence 1: Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.
  Fact 1.0: Billie Joe Armstrong was born on February 17, 1972.
  Fact 1.1: Billie Joe Armstrong is an American musician, singer, songwriter and actor.
  Fact 1.2: Billie Joe Armstrong is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day.
  Fact 1.3: Billie Joe Armstrong co-founded Green Day with Mike Dirnt.

Retrieved items:
<item 1>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 1>

<item 2>
Green Day related projects
Since 1991, some members of punk band Green Day have branched out past their "main band" and have started other projects with other musicians and have released
full-length albums and several EPs.

 Notable related projects of Green Day include Billie Joe Armstrong's Pinhead Gunpowder (which also featured Green Day's other guitarist Jason White), The
Frustrators in which Mike Dirnt plays bass, and The Network which many speculate has all three members of Green Day, although under stagenames.
</item 2>

<item 3>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 3>

<item 4>
Graph query row:
group: Pinhead Gunpowder
relationship: MEMBER_OF
person: Billie Joe Armstrong
</item 4>

<item 5>
Bill Schneider (musician)
William 'Bill' Schneider (born June 17) is an American musician.

 He has played bass in the punk rock band Pinhead Gunpowder since 1990, as well as providing vocals and writing the lyrics to the song "Backyard Flames".

 He has also been in many other bands such as The Influents, Monsula and Uranium 9v, Sawhorse and The Skinflutes.

 Bill was worked as bass tech during the Nimrod era for Mike Dirnt, Green Day's and as guitar tech on their albums "Nimrod" and "Warning", American Idiot, 21st
Century Breakdown, Uno Dos Tres, Revolution Radio and was their day to day manager from 2003-2012 for their "American Idiot" and 21st Century Breakdown tours
through Uno Dos Tres.

 He also owned "Broken Guitars", a guitar shop in Oakland, California with Billie Joe Armstrong.

 Bill Still works on the sidelines with the band as well as in the studio.
</item 5>

<item 6>
Pinhead Gunpowder (2008 EP)
Pinhead Gunpowder (also known by fans as the West Side Highway EP) is the second eponymously titled extended play by the American punk rock band Pinhead
Gunpowder.

 It was released on August 19, 2008, through Recess Records, the group's first release on Recess.

 "Pinhead Gunpowder" marked the band's first new release in over 5 years.

 The first pressings of the extended play were on green and blue vinyl.

 All of the tracks were rereleased on the 2009 compilation "Kick Over the Traces".
</item 6>

<item 7>
Compulsive Disclosure
Compulsive Disclosure is the second compilation album by the American punk rock band Pinhead Gunpowder.

 It was released on October 21, 2003, through Lookout!

 Records.

 The album features songs from the group's eponymously titled 2000 EP, the "Dillinger Four / Pinhead Gunpowder" split EP, the "8 Chords, 328 Words" EP and also
features re-recorded versions of the songs "2nd Street" and "At Your Funeral" (originally from "Dillinger Four / Pinhead Gunpowder").

 "Compulsive Disclosure" was re-released on CD and vinyl through Recess Records on February 12, 2010, with two unreleased tracks, "Salting Agents" and "El Lasso
Grappo".
</item 7>

<item 8>
Billie Joe Armstrong
Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter,
and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.

 He is also a guitarist and vocalist for the punk rock band Pinhead Gunpowder and provides lead vocals for Green Day's side projects Foxboro Hot Tubs and The
Network.
</item 8>

<item 9>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 9>

<item 10>
Carry the Banner
Carry the Banner is the third EP by the Berkeley, California-based punk rock band Pinhead Gunpowder.

 Originally released on 10" vinyl in December 1994 through Too Many Records, the EP was reissued on CD by Lookout Records shortly after as the initial vinyl
pressing sold out quickly.

 It was the group's first release to feature Jason White on guitar/vocals, replacing Sarah Kirsch, who left the band in 1994 due to differences with Billie Joe
Armstrong after his main band Green Day signed to major label Reprise Records.
</item 10>
=== END ITEM id=bb27521b42db68b1 ===

=== ITEM id=4c2075832e959a60 ===
Gold sentences and facts:
Sentence 0: Essence is a monthly magazine for African American women between the ages of 18 and 49.
  Fact 0.0: Essence is a monthly magazine.
  Fact 0.1: Essence is for African American women.
  Fact 0.2: Essence is aimed at women between the ages of 18 and 49.
Sentence 1: ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
  Fact 1.0: ALT for Damerne is a women's magazine.
  Fact 1.1: ALT for Damerne means 'All for the Ladies' in English.
  Fact 1.2: ALT for Damerne is written in the Danish language.
  Fact 1.3: ALT for Damerne is a weekly magazine.
  Fact 1.4: ALT for Damerne is published in Copenhagen, Denmark.

Retrieved items:
<item 1>
Makedonsko Sonce
Makedonsko Sonce (Macedonian: ; English: Macedonian Sun ) is a weekly magazine published in the Republic of Macedonia.

 The title means "Macedonian Sun", referring to the Vergina Sun, which is used in the logo of the magazine.

 It was established by Gjorgija (George) Atanasoski and the first edition was published on June 24, 1994.

 It describes its own political stance as affirming "Macedonian national values, not only on territory of Macedonia but also in the other parts of ethnic
Macedonia".
</item 1>

<item 2>
Alt for Damerne
ALT for Damerne (meaning "All for the Ladies" in English) is a Danish language weekly women's magazine published in Copenhagen, Denmark.
</item 2>

<item 3>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 3>

<item 4>
Lene Thiesen
Lene Thiesen (b. 1947 in Copenhagen) is the former director of the Copenhagen International Theatre (Danish: "Københavns Internationale Teater or KIT" ).

 In 1992, Thiesen was named 'Woman of the Year' by the Danish magazine "Alt for damerne".
</item 4>

<item 5>
Essence Music Festival
The Essence Festival, known as "the party with a purpose", is an annual music festival which started in 1995 as a one-time event to celebrate the 25th
anniversary of "Essence", a magazine aimed primarily towards African-American women.

 It is the largest event celebrating African-American culture and music in the United States.

 Locally referred to as the Essence Fest, it has been held in New Orleans, Louisiana every year since 1994 except for 2006, when it was held in Houston, Texas
due to Hurricane Katrina's effect on New Orleans.

 It also held in Durban, KwaZulu-Natal in 2016.

 It features artists simultaneously performing on a main stage as well as four standing-room only superlounge stages.
</item 5>

<item 6>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 6>

<item 7>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 7>

<item 8>
Tea Moderna (magazine)
Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
</item 8>

<item 9>
Scott Poulson-Bryant
Scott Poulson-Bryant is an award-winning American journalist and author.

 One of the co-founding editors of Vibe magazine in 1992 (and the editor who gave the magazine its name), Poulson-Bryant's journalism, profiles, reviews, and
essays have appeared in such publications as the "New York Times", "the Village Voice", "Rolling Stone", "Spin", "Essence", "Ebony", and "The Source".

 He is the author of "HUNG: A Meditation on the Measure of Black Men in America" (published by Doubleday Books in 2006) and a novel called "The VIPs".
</item 9>

<item 10>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 10>
=== END ITEM id=4c2075832e959a60 ===

=== ITEM id=22111019c9a6f64f ===
Gold sentences and facts:
Sentence 0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie, Michigan.
  Fact 0.0: The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant.
  Fact 0.1: The Saint Marys Falls Hydropower Plant is a Cloverland Electric Cooperative hydroelectric plant.
  Fact 0.2: The Saint Marys Falls Hydropower Plant is in Sault Ste. Marie, Michigan.
Sentence 1:  Excavation of the power canal began in September 1898 and was completed in June 1902.
  Fact 1.0: Excavation of the Edison Sault Power Canal began in September 1898.
  Fact 1.1: Excavation of the Edison Sault Power Canal was completed in June 1902.
Sentence 2:  The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
  Fact 2.0: The canal was named a Historic Civil Engineering Landmark in 1983.
  Fact 2.1: The hydroelectric complex was named a Historic Civil Engineering Landmark in 1983.
Sentence 3: The Cayuga–Seneca Canal is a canal in New York, United States.
  Fact 3.0: The Cayuga–Seneca Canal is a canal in New York, United States.
Sentence 4:  It is now part of the New York State Canal System.
  Fact 4.0: The Cayuga–Seneca Canal is now part of the New York State Canal System.

Retrieved items:
<item 1>
Power canal
A Power Canal refers to a canal used for hydraulic power generation, rather than for transport of watercraft.

 The power canal was a major factor in the Industrial revolution in New England in the 19th century.

 Most early power canals were mill races used mechanically to transfer power directly from falling water to machinery in mill buildings.

 Later, the hydraulic power generated electricity locally for the same mill factories.

 These power canals were often filled in as electricity (transported by power lines) replaced the need for local water power, and road transport needs or city
expansion needs reclaimed the land.

 Some hydraulic power canals were transformed into local electric generators, but most were closed.

 Remains of power canals can be seen in old mill towns and are often protected as historical structures today.
</item 1>

<item 2>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 2>

<item 3>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 3>

<item 4>
Erie Canal Lock 52 Complex
Erie Canal Lock 52 Complex is a national historic district located at Port Byron and Mentz in Cayuga County, New York.

 The district includes two contributing buildings (the Erie House and the blacksmith shop / mule barn); three contributing engineering structures (Erie Canal
Lock 52, culvert, and canal prism of the enlarged Erie Canal); and archaeological sites associated with the canal operations.

 Lock 52 was constructed 1849-1853 as part of the Enlarged Erie Canal program.

 It remained in operation until the rerouting of the canal under the New York State Barge Canal System in 1917.

 The Erie House was built in 1894 and is a two story frame structure that housed a saloon and hotel.
</item 4>

<item 5>
Edison Sault Power Canal
The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie,
Michigan.

 Excavation of the power canal began in September 1898 and was completed in June 1902.

 The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
</item 5>

<item 6>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 6>

<item 7>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 7>

<item 8>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 8>

<item 9>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 9>

<item 10>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 10>
=== END ITEM id=22111019c9a6f64f ===

=== ITEM id=4f63f6fc3630fc6a ===
Gold sentences and facts:
Sentence 0: Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.
  Fact 0.0: Edward Sharpe and the Magnetic Zeros is an American folk rock band.
  Fact 0.1: Edward Sharpe and the Magnetic Zeros was formed in Los Angeles, California.
  Fact 0.2: Edward Sharpe and the Magnetic Zeros was formed in 2007.
Sentence 1: Louis XIV is an American rock band from San Diego, California.
  Fact 1.0: Louis XIV is an American rock band.
  Fact 1.1: Louis XIV is from San Diego, California.
Sentence 2:  The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.
  Fact 2.0: The band released four EPs between 2003 and 2007.
  Fact 2.1: The band released three albums between 2003 and 2008.
  Fact 2.2: Two of the band's three albums, the latter two, were distributed by Atlantic Records.

Retrieved items:
<item 1>
Edward Sharpe and the Magnetic Zeros
Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.

 The group is led by lead singer Alex Ebert.

 The band's name is based on a story Ebert wrote in his youth, about a messianic figure named Edward Sharpe.

 Drawing from roots rock, folk, gospel, and psychedelic music, the band's image and sound evoke the hippie movement of the 1960s and 1970s.

 The group's first show was played July 18, 2007 at The Troubadour in West Hollywood, California.

 Their first studio album, "Up from Below", was released on July 7, 2009 on Community Records and featured the popular single "Home".

 The group released their second full-length album, "Here", on May 29, 2012, and third album, "Edward Sharpe and the Magnetic Zeros", on July 23, 2013.

 Their fourth studio album, "PersonA", was released in April 2016.
</item 1>

<item 2>
Alexander (Alex Ebert album)
Alexander is the first solo album by American musician Alex Ebert, lead singer of Edward Sharpe and the Magnetic Zeros and Ima Robot.

 It was released on March 1, 2011 under the Community Records label.

 The song "Truth" was featured in the season 4 premiere of AMC's "Breaking Bad" titled "Box Cutter".
</item 2>

<item 3>
Edward Sharpe and the Magnetic Zeros (album)
Edward Sharpe and the Magnetic Zeros is the third album by Edward Sharpe and the Magnetic Zeros.

 It was released on July 23, 2013 in North America and was released on July 29, 2013 around the world through Vagrant Records, Rough Trade Records and Communion
Records.

 Frontman Alex Ebert stated that "These songs mean everything to me - It's the rawest, most liberated, most rambunctious stuff we've done."
</item 3>

<item 4>
PersonA
PersonA (pronounced "Persona") is the fourth studio album by the band Edward Sharpe and the Magnetic Zeros.

 It was released on April 15, 2016.
</item 4>

<item 5>
Orpheo McCord
Orpheo McCord (born 1979) is an American drummer and percussionist.

 He is an original member of Edward Sharpe and the Magnetic Zeros, and a former member of The Fall, Fool's Gold and Cass McCombs.
</item 5>

<item 6>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 6>

<item 7>
Subatomic Sound System
Founded in 1999 by John Emch and Noah Shachtman, Subatomic Sound System brought together musicians, producers, DJs and visual artists from a variety of
backgrounds and traditions primarily based in New York City and Brooklyn to form a record label and collective that built on a combination of new music
technology and traditional instruments to produce music across a variety of genres, often combining genres, in an effort to adapt 1970s’ Jamaican sound system
culture and dub studio techniques to current music genres and forms of live performance.

 In fall 2008, Subatomic Sound System garnered international attention for a limited edition vinyl 12" featuring their collaboration with Vienna's
Dubblestandart and dub inventor Lee "Scratch" Perry, releasing the first songs from Perry in the dubstep genre, one of the first recorded examples of a tangible
connection between the popular UK based electronic genre that emerged in the begin of the first decade of the 21st century and the Jamaican dub from the 1970s
where dubstep's origins were rooted and which had been primarily originated by Perry himself.

 Beginning in 2008, Subatomic Sound System started hosting weekly radio shows on 91.5fm, Radio New York and webcast on Brooklyn Radio.
</item 7>

<item 8>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 8>

<item 9>
Louis XIV (band)
Louis XIV is an American rock band from San Diego, California.

 The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.

 The band broke up in 2009, but in 2013, in an interview with The Reno Dispatch, Jason Hill confirmed that the band had decided to reunite.
</item 9>

<item 10>
Lingdian
Zero Band () are a Chinese rock band from Inner Mongolia.

 The lead singer is Zhou Xiaoou (周晓鸥).

 The English name "Zero Band" was used in Beijing concert advertisements, though the band did not originally use an English name; the name has also been
translated as "Point Zero" in some older English publications.

 The band appeared on the Jingwen label in China, but were licensed outside China by JVC.

 One of the top Beijing bands at the end of the 1990s, Lingdian originally suffered the same resistance from government-owned venues as other indigenous PRC
rock bands.
</item 10>
=== END ITEM id=4f63f6fc3630fc6a ===

=== ITEM id=f80527980e655176 ===
Gold sentences and facts:
Sentence 0: Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris Institute or Art School, Harris Technical School and the Harris Orphanage.
  Fact 0.0: Edmund Robert Harris lived c. 1804 to 27 May 1877.
  Fact 0.1: Edmund Robert Harris was an English lawyer.
  Fact 0.2: Edmund Robert Harris was from Preston, Lancashire, UK.
  Fact 0.3: Edmund Robert Harris was the principal benefactor of the Harris Museum.
  Fact 0.4: Edmund Robert Harris was the principal benefactor of the Harris Institute or Art School.
  Fact 0.5: Edmund Robert Harris was the principal benefactor of the Harris Technical School.
  Fact 0.6: Edmund Robert Harris was the principal benefactor of the Harris Orphanage.
Sentence 1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
  Fact 1.0: The Harris Museum, Art Gallery & Preston Free Public Library is a museum building in Preston.
  Fact 1.1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed building.

Retrieved items:
<item 1>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 1>

<item 2>
Edmund de Unger
Edmund Robert Anthony de Unger (Hungarian: "Odon Antal Robert de Unger" , b 6 August 1918, Budapest - d 25 January 2011, Ham, Surrey) was a Hungarian-born
property developer and art collector.

 In London he built up the Keir Collection, one of the greatest post-war collections of Islamic art, bequeathed in 2008 to the Pergamon Museum of Islamic Art in
Berlin.

 The arrangement for the museum to curate the collection came to an end in July 2012.

 The collection is now hosted by the Dallas Museum of Art as of May 2014 for a 15-year renewable loan.
</item 2>

<item 3>
Hanbys Corner, Delaware
The last Hanby to occupy the property, Albert T. Hanby (1881-1947), another son of Samuel, attended West Chester State College before getting his law degree
from Penn Law School.

 Albert became a Philadelphia lawyer and left his farm at Hanby's Corner to be used for the good of "all the children in Delaware".

 He and his wife created a foundation in 1945 to protect the property from further development.

 Today the YMCA operates their Hanby Camp there, and the trust provide scholarships for kids who might not otherwise be able to attend.

 In 2007 that support was over $70,000.
</item 3>

<item 4>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 4>

<item 5>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 5>

<item 6>
Edmund Robert Harris
Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris
Institute or Art School, Harris Technical School and the Harris Orphanage.
</item 6>

<item 7>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 7>

<item 8>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 8>

<item 9>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 9>

<item 10>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 10>
=== END ITEM id=f80527980e655176 ===

=== ITEM id=86efa5eacd4d1c57 ===
Gold sentences and facts:
Sentence 0: While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and Zoot Sims.
  Fact 0.0: The musician studied at the University of Michigan.
  Fact 0.1: While studying at the University of Michigan, the musician played with visiting musician Dizzy Gillespie.
  Fact 0.2: While studying at the University of Michigan, the musician played with visiting musician Miles Davis.
  Fact 0.3: While studying at the University of Michigan, the musician played with visiting musician Thad Jones.
  Fact 0.4: While studying at the University of Michigan, the musician played with visiting musician Billy Mitchell.
  Fact 0.5: After that, the musician played with Sonny Stitt, Count Basie and Al McKibbon.
  Fact 0.6: After that, the musician played with Cannonball Adderley, Percy Heath, Philly Joe Jones and Lou Donaldson.
  Fact 0.7: After that, the musician played with Donald Byrd, Kenny Dorham and Zoot Sims.
Sentence 1: Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.
  Fact 1.0: Miles Dewey Davis III was born on May 26, 1926.
  Fact 1.1: Miles Dewey Davis III died on September 28, 1991.
  Fact 1.2: Miles Dewey Davis III was an American jazz trumpeter.
  Fact 1.3: Miles Dewey Davis III was an American jazz bandleader.
  Fact 1.4: Miles Dewey Davis III was an American jazz composer.

Retrieved items:
<item 1>
Don Ellis
Donald Johnson Ellis (July 25, 1934 – December 17, 1978) was an American jazz trumpeter, drummer, composer, and bandleader.

 He is best known for his extensive musical experimentation, particularly in the area of time signatures. Later in his life he worked as a film composer,
contributing a score to 1971's "The French Connection" and 1973's "The Seven-Ups".
</item 1>

<item 2>
Louis Smith (musician)
While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before
going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and
Zoot Sims.
</item 2>

<item 3>
Paul Smith (defensive end)
Paul Edward Smith (August 13, 1945 – March 14, 2000) was a collegiate and professional American football defensive end.

 Smith was selected in the 1968 Common Draft by the American Football League's Denver Broncos and played in the National Football League for the Broncos and
Washington Redskins.
</item 3>

<item 4>
Miles Davis
Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.

 He is among the most influential and acclaimed figures in the history of jazz and 20th century music.

 Davis adopted a variety of musical directions in his five-decade career which kept him at the forefront of a number of major stylistic developments in jazz.
</item 4>

<item 5>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 5>

<item 6>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 6>

<item 7>
Count Basie
William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.

 His mother taught him to play the piano and he started performing in his teens.

 Dropping out of school, he learned to operate lights for vaudeville and to improvise accompaniment for silent films at a local movie theater in his home town
of Red Bank, New Jersey.

 By age 16, he increasingly played jazz piano at parties, resorts and other venues.

 In 1924, he went to Harlem, where his performing career expanded; he toured with groups to the major jazz cities of Chicago, St. Louis and Kansas City.

 In 1929 he joined Bennie Moten's band in Kansas City, and played with them until Moten's death in 1935.
</item 7>

<item 8>
Thad Jones
Thaddeus Joseph Jones (March 28, 1923 – August 20, 1986) was an American jazz trumpeter, composer, and bandleader who has been called "one of the all-time
greatest jazz trumpet soloists."
</item 8>

<item 9>
Marty Marsala
Marty Marsala (2 April 1909 – 27 April 1975) was an American jazz trumpeter born in Chicago, perhaps best known for working from 1926-1946 with his brother Joe
Marsala in a big band in New York City and Chicago.

 He had also toured with various artists, such as Chico Marx and Miff Mole, to name a few.

 During the 1940s Marsala was a celebrated West Coast jazz trumpeter, commuting back and forth from Chicago to San Francisco frequently.

 In various club settings Marsala shared stages with Earl Hines and Sidney Bechet.
</item 9>

<item 10>
Louis Smith (gymnast)
Louis Antoine Smith, MBE (born 22 April 1989) is a British artistic gymnast.

 He received a bronze medal and a silver medal on the pommel horse at the 2008 Beijing Olympics and the 2012 London Olympics respectively, with the former
marking the first time a British gymnast had placed in an Olympic event since 1928.

 He followed this up with a second consecutive silver medal on the pommel horse at the 2016 Rio Olympics, finishing behind teammate Max Whitlock.

 Smith was part of the Great Britain team that took the bronze in the men's artistic team all-around at the 2012 London Olympics.

 He is the only British gymnast to win Olympic medals in three separate Games.

 In 2015 he became the European champion on Pommel Horse.
</item 10>
=== END ITEM id=86efa5eacd4d1c57 ===

=== ITEM id=063e253a74ed2348 ===
Gold sentences and facts:
Sentence 0: Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and "".
  Fact 0.0: Steve Rash is an American film director.
  Fact 0.1: Steve Rash is an American producer.
  Fact 0.2: Steve Rash is best known for directing The Buddy Holly Story.
  Fact 0.3: Steve Rash is best known for directing Can't Buy Me Love.
  Fact 0.4: Steve Rash is best known for directing Queens Logic.
  Fact 0.5: Steve Rash is best known for directing two further films whose titles are blank in the source text.
Sentence 1: Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.
  Fact 1.0: Can't Buy Me Love is a 1987 American film.
  Fact 1.1: Can't Buy Me Love is a teen romantic comedy feature film.
  Fact 1.2: Can't Buy Me Love stars Patrick Dempsey.
  Fact 1.3: Can't Buy Me Love stars Amanda Peterson.
  Fact 1.4: The story of Can't Buy Me Love is about a nerd at a high school in Tucson, Arizona.
  Fact 1.5: In Can't Buy Me Love, the nerd gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

Retrieved items:
<item 1>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 1>

<item 2>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 2>

<item 3>
Heavyweights
Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.

 The film centers around a fat camp for kids that is taken over by a fitness guru named Tony Perkis (Ben Stiller).
</item 3>

<item 4>
Bring It On: In It to Win It
Bring It On: In It to Win It is a 2007 teen film directed by Steve Rash and starring Ashley Benson, Cassie Scerbo and Michael Copon.
</item 4>

<item 5>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 5>

<item 6>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 6>

<item 7>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 7>

<item 8>
Steve Rash
Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and
"".
</item 8>

<item 9>
Just Friends
Just Friends is a 2005 American romantic comedy film directed by Roger Kumble, written by Adam 'Tex' Davis and starring Ryan Reynolds, Amy Smart, Anna Faris,
Chris Klein and Christopher Marquette.

 The plot focuses on a formerly overweight high school nerd (Reynolds) who attempts to free himself from the friend zone after reconnecting with his lifelong
crush and best friend (Smart) while visiting his hometown for Christmas.
</item 9>

<item 10>
Can't Buy Me Love (film)
Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in
Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

 The film was directed by Steve Rash and takes its title from a Beatles song with the same title.
</item 10>
=== END ITEM id=063e253a74ed2348 ===

=== ITEM id=e9cae61d0769e004 ===
Gold sentences and facts:
Sentence 0: The Straight Story is a 1999 internationally co-produced biographical road drama film directed by David Lynch.
  Fact 0.0: The Straight Story is a 1999 film.
  Fact 0.1: The Straight Story is internationally co-produced.
  Fact 0.2: The Straight Story is a biographical road drama film.
  Fact 0.3: The Straight Story was directed by David Lynch.
Sentence 1: Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.
  Fact 1.0: Frozen is a 2013 film.
  Fact 1.1: Frozen is American.
  Fact 1.2: Frozen is a 3D computer-animated film.
  Fact 1.3: Frozen is a musical fantasy film.
  Fact 1.4: Frozen was produced by Walt Disney Animation Studios.
  Fact 1.5: Frozen was released by Walt Disney Pictures.

Retrieved items:
<item 1>
A broad film, television, documentary, and industry collection spans silent-era cinema through 2010, covering action, comedy, drama, horror, musicals, noir, and
more. Highlights include *The Razor’s Edge*, *The Little Foxes*, *Die Hard*, *Unbreakable*, *Signs*, *The Straight Story*, *West Side Story*, and *Rogue One*.
Documentaries address music, war, politics, disability, and culture. Profiles feature Bruce Willis, David
</item 1>

<item 2>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 2>

<item 3>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 3>

<item 4>
The entries span adult animation, comedy, drama, westerns, science fiction, family films, and television. Ralph Bakshi’s *Fritz the Cat* (1972), the first U.S.
animated feature rated X, and *Fire and Ice* (1983) contrast with his live-action *Cool and the Crazy*. Disney titles include *The Apple Dumpling Gang* films,
*Darby O’Gill*, and *Home on the Range*. Other franchises include *The Naked
</item 4>

<item 5>
The Straight Story
The Straight Story is a 1999 internationally co-produced biographical road drama film directed by David Lynch.

 The film was edited and produced by Mary Sweeney, Lynch's longtime partner and co-worker.

 She co-wrote the script with John E. Roach.

 The film is based on the true story of Alvin Straight's 1994 journey across Iowa and Wisconsin on a lawn mower.

 Alvin (Richard Farnsworth) is an elderly World War II veteran who lives with his daughter Rose (Sissy Spacek), a kind woman with an intellectual disability.

 When he hears that his estranged brother Lyle (Harry Dean Stanton) has suffered a stroke, Alvin makes up his mind to go visit him and hopefully make amends
before he dies.

 Because Alvin's legs and eyes are too impaired for him to receive a driving license, he hitches a trailer to his recently purchased thirty-year-old John Deere
110 Lawn Tractor, having a maximum speed of about 5 miles per hour, and sets off on the 240 mile journey from Laurens, Iowa to Mount Zion, Wisconsin.
</item 5>

<item 6>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 6>

<item 7>
The entries span film, art, and music: *Cool and the Crazy* (1994) is Ralph Bakshi’s first live-action feature, about an unfaithful 1950s couple; *No eres tú,
soy yo* (2010) is a Mexican remake of an Argentine comedy; and *The Straight Story* (1999) follows Alvin Straight’s 240-mile tractor journey to visit his
estranged, ill brother. Other films include *Lilly the Witch
</item 7>

<item 8>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 8>

<item 9>
The Blizzard (1923 film)
The Blizzard is a 1923 Swedish drama film directed by Mauritz Stiller, starring Einar Hanson, Mary Johnson, Pauline Brunius and Hugo Björne.

 The film's original Swedish title is Gunnar Hedes saga, which means "The story of Gunnar Hede".

 The narrative revolves a student who tries to save his family's mansion which is facing bankruptcy.

 The film is loosely based on the Selma Lagerlöf novel "The Tale of a Manor".
</item 9>

<item 10>
The entries survey European films and music drama, 1913–1961. Swedish director Mauritz Stiller’s works include *The Blizzard* (a student saves his family
estate), *Erotikon* (a professor’s wife attracts suitors), and *The Saga of Gösta Berling*, Greta Garbo’s breakout film. Hungarian director Béla Gaál made
comedies including *The Dream Car*. Other highlights: *Triumph of the Will*, Riefenstahl’s
</item 10>
=== END ITEM id=e9cae61d0769e004 ===

=== ITEM id=ef79305ce2266599 ===
Gold sentences and facts:
Sentence 0:  It has an area of 1208 km2 and had a population of 600,000 in 2003.
  Fact 0.0: Ezhou has an area of 1208 km2.
  Fact 0.1: Ezhou had a population of 600,000 in 2003.
Sentence 1: Ezhou () is a prefecture-level city in eastern Hubei Province, China.
  Fact 1.0: Ezhou is a prefecture-level city.
  Fact 1.1: Ezhou is in eastern Hubei Province.
  Fact 1.2: Ezhou is in China.
Sentence 2:  As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.
  Fact 2.0: As of the 2010 census, Ezhou had a population of 1,048,668.
  Fact 2.1: As of the 2010 census, 668,727 of Ezhou's residents lived in the core Echeng District.
Sentence 3:  The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
  Fact 3.0: The Ezhou-Huanggang built-up (or metro) area was home to 1,035,496 inhabitants.
  Fact 3.1: The inhabitants of the Ezhou-Huanggang built-up area came from the Echeng and Huangzhou districts of Huanggang.

Retrieved items:
<item 1>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 1>

<item 2>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 2>

<item 3>
Dafeng, Yancheng, is a coastal Jiangsu district with 112 km of shoreline, a salt-making history, protected ecosystems, and China’s largest Père David’s deer
reserve. Changle, Fuzhou, Fujian, has extensive land and sea areas, Tang-era origins, and became a district in 2017. Fuzhoucheng is a Liaoning town near
highways, a reservoir, and a bird reserve. Jiangyin dialect is a Taihu Wu
</item 3>

<item 4>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 4>

<item 5>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 5>

<item 6>
Jiangyin dialect
Jiangyin dialect (江阴话) is a Northern Wu Chinese dialect spoken in the city of Jiangyin in Jiangsu province.

 Jiangyin dialect is a member of the Wu Chinese Taihu Wu family of dialects, which means the inhabitants speak a dialect similar to that of nearby Wuxi,
Changzhou, Suzhou, and Shanghai.

 Jiangyin dialect itself is of the Piling variety, related to the Changzhou dialect.

 Jiangyin dialect has the highest degree of mutual intelligibility with the dialects of the closest neighboring cities of Changzhou and Wuxi but also has a
fairly large degree of mutual intelligibility with the dialects of nearby Suzhou and Shanghai.

 As one travels south towards Wuxi away from the urban center of Jiangyin, Jiangyin dialect gradually becomes more and more closer sounding to the Wuxi dialect.
</item 6>

<item 7>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 7>

<item 8>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 8>

<item 9>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 9>

<item 10>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 10>
=== END ITEM id=ef79305ce2266599 ===

=== ITEM id=1c985f852db3167c ===
Gold sentences and facts:
Sentence 0: The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).
  Fact 0.0: The Washington City Canal operated from 1815 until the mid-1850s.
  Fact 0.1: The Washington City Canal operated in Washington, D.C.
  Fact 0.2: The Washington City Canal connected the Anacostia River to Tiber Creek.
  Fact 0.3: The Washington City Canal connected the Anacostia River to the Potomac River.
  Fact 0.4: The Washington City Canal later connected the Anacostia River to the Chesapeake and Ohio Canal (C&O).
  Fact 0.5: The Anacostia River was called the 'Eastern Branch' at that time.
Sentence 1: The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River to Hagerstown, Indiana.
  Fact 1.0: The Whitewater Canal was built between 1836 and 1847.
  Fact 1.1: The Whitewater Canal spanned a distance of seventy-six miles.
  Fact 1.2: The Whitewater Canal stretched from Lawrenceburg, Indiana.
  Fact 1.3: Lawrenceburg, Indiana is on the Ohio River.
  Fact 1.4: The Whitewater Canal stretched to Hagerstown, Indiana.

Retrieved items:
<item 1>
Tiber Creek
Tiber Creek or Tyber Creek is a tributary of the Potomac River in Washington, D.C. In the 19th century it was modified to become part of the Washington City
Canal, and subsequently was enclosed in a tunnel.
</item 1>

<item 2>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 2>

<item 3>
Cincinnati and Whitewater Canal Tunnel
The Cincinnati and Whitewater Canal Tunnel is a historic yet abandoned canal tunnel in the southwestern corner of the U.S. state of Ohio.

 Located within the village of Cleves near Cincinnati, it was constructed in 1837 as part of the Whitewater Canal system.

 Since the canal's closure, it has largely been forgotten, but it has been designated a historic site.
</item 3>

<item 4>
Whitewater Canal
The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River
to Hagerstown, Indiana.
</item 4>

<item 5>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 5>

<item 6>
Broad Street Bridge (Rochester, New York)
Erie Canal: Second Genesee Aqueduct, also known as the Broad Street Aqueduct or Broad Street Bridge, is a historic stone aqueduct located at Rochester in Monroe
County, New York.

 It was constructed in 1836-1842 and originally carried the Erie Canal over the Genesee River.

 The overall length of the aqueduct including the wings and abutments is 800 ft .

 The aqueduct is 70 ft wide and has massive parapets on either side.

 It is one of four major aqueducts in the mid-19th century Erie Canal system.

 In 1927, a roadbed was added to carry automobile traffic and named Broad Street.

 It also carried a part of the Rochester Subway.
</item 6>

<item 7>
Tiber Island (District of Columbia)
Tiber Island was a man-made island in Washington, D.C. formed when the Washington City Canal was dug to connect the stream beds of Tiber Creek and James Creek,
creating an island out of an existing peninsula southwest of the Capitol.

 The canals have since been filled in, rejoining the island to the mainland.

 The Southwest Waterfront, Buzzard Point, National Mall, and L'Enfant Plaza areas were once on the island; at that time, their isolation from "the mainland" led
to the area's colloquial nickname as "The Island."
</item 7>

<item 8>
Whitewater River (Great Miami River)
The Whitewater River is a 101 mi southerly flowing right tributary of the Great Miami River in southeastern Indiana and southwestern Ohio in the United States.

 It is formed by the confluence of two forks, the West Fork and East Fork.

 The name is a misnomer, as there is no true white water on the river.

 However, there are many rapids due to the steep gradient present - the river falls an average of 6 ft/mile .

 The gradient rendered upstream navigation impossible, and in the mid-nineteenth century resulted in the construction of the Whitewater Canal paralleling the
river from north of Connersville, Indiana, to the Ohio River.
</item 8>

<item 9>
Washington City Canal
The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at
that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).

 The canal fell into disuse in the late 19th century and the city government covered over or filled in various sections.
</item 9>

<item 10>
Duck Creek Aqueduct
Duck Creek Aqueduct, also known as the Metamora Aqueduct and Whitewater Canal Aqueduct, is a historic aqueduct carrying the Whitewater Canal over Duck Creek in
Metamora Township, Franklin County, Indiana.

 It is the only surviving covered wood aqueduct in the United States as such it is a form of covered bridge.

 It is a single-span Burr through truss aqueduct and measures approximately 90 feet long, 25 feet wide, and 25 feet deep.

 The original aqueduct was built between 1839 and 1843.

 The present structure was built in 1846, after the original aqueduct was washed out in a flood.

 It was strengthened in 1868, and repaired in 1901.

 After abandonment and deterioration, the Duck Creek Aqueduct was restored to its present appearance in 1946-1949.
</item 10>
=== END ITEM id=1c985f852db3167c ===
