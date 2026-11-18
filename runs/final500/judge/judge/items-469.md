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

=== ITEM id=0cc7a1720416cbab ===
Gold sentences and facts:
Sentence 0:  Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great America).
  Fact 0.0: The roller coaster was built in 1988 by the TOGO company.
  Fact 0.1: The roller coaster built by the TOGO company was the fastest roller coaster in the world when it was built.
  Fact 0.2: The roller coaster built by the TOGO company took the record for fastest roller coaster in the world from American Eagle at Six Flags Great America.
Sentence 1: TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky cycles and other amusement rides.
  Fact 1.0: TOGO (株式会社トーゴ, Kabushiki-gaisha Tōgo) was a Japanese amusement ride company.
  Fact 1.1: TOGO built roller coasters.
  Fact 1.2: TOGO built giant wheels.
  Fact 1.3: TOGO built carousels.
  Fact 1.4: TOGO built flumes.
  Fact 1.5: TOGO built dark rides.
  Fact 1.6: TOGO built sky cycles.
  Fact 1.7: TOGO built other amusement rides.

Retrieved items:
<item 1>
List of roller coaster rankings
Roller coasters are amusement rides developed for amusement parks and modern theme parks.

 During the 16th and 17th centuries, rides consisting of wooden sleds that took riders down large slides made from ice were popular in Russia.

 The first roller coasters, where the train was attached to a wooden track, first appeared in France in the early 1800s.

 Although wooden roller coasters are still being produced, steel roller coasters are more common and can be found on every continent except Antarctica.
</item 1>

<item 2>
Bandit (Yomiuriland)
Bandit (バンデット , Bandetto ) is a steel roller coaster located at Yomiuriland in the city of Inagi, near Tokyo, Japan.

 Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great
America).

 It lost the record to Magnum XL-200 at Cedar Point one year later.
</item 2>

<item 3>
Giovanola
Giovanola Freres SA was one of the prominent thrill ride manufacturers in the world.

 It was well known for thrill rides and also built electrical power stations, water storage tanks, pipelines, highway bridges, and many other steel products.

 The company started out as a small metal forging shop, founded by Joseph Giovanola in 1888.

 It served as a subcontractor to Intamin supplying rides and roller coasters.

 In 1998 Giovanola started marketing directly under the name Giovanola Amusement Rides Worldwide.

 The company was based in Monthey, Switzerland.

 During its last years in business, Giovanola fashioned steel behind the scenes for companies such as Intamin and Bolliger & Mabillard.

 The company also built its own thrill rides and roller coasters from 1998 to 2001.

 Examples of Giovanola roller coasters are Goliath at Six Flags Magic Mountain, Titan at Six Flags Over Texas and Anaconda at South Africa's Gold Reef City.
</item 3>

<item 4>
Pretzel Amusement Ride Company
The Pretzel Amusement Ride Company was a famous manufacturer of pretzel dark rides. Pretzel built over 1400 pretzel rides and sold them to carnivals and parks.

 The pretzel ride was invented by Marvin Rempfer.

 Leon Cassidy was Marvin's partner in the Pretzel Amusement Ride Company.

 Both names are on the patent.

 Leon Cassidy patented the single-rail dark ride in 1928 along with Marvin Rempfer.

 The company originated in Tumbling Dam Park on the banks of Sunset Lake in Bridgeton, New Jersey.

 A rider said that ""It felt like I was turned and twisted like a Pretzel"", so the name Pretzel was chosen.

 A large heavy pretzel design was originally affixed to the front of each car to prevent the car from flipping backwards.

 In 1929, a standard Pretzel ride had five cars, 350 feet of track, and was one and a half minutes per ride.

 A pretzel ride sold for $1,200.
</item 4>

<item 5>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 5>

<item 6>
TOGO
TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky
cycles and other amusement rides.
</item 6>

<item 7>
Millennium Force
Millennium Force is a steel roller coaster built by Intamin at Cedar Point amusement park in Sandusky, Ohio, United States.

 It was the fourteenth roller coaster to be built at the park since Blue Streak opened in 1964.

 Upon completion in 2000, Millennium Force broke six world records and was the world's first Giga Coaster, a roller coaster that exceeds 300 ft in height and
completes a full circuit.

 It was briefly the tallest and fastest in the world until Steel Dragon 2000 opened later the same year.

 The ride is also the third-longest roller coaster in North America after The Beast at Kings Island and Fury 325 at Carowinds.

 It was the first roller coaster to use a cable lift system rather than a traditional chain lift.

 The coaster has a 310 ft , 45-degree lift hill with a 300 ft drop and features two tunnels, three overbanked turns, and four hills.

 It has a top speed of 93 mph .
</item 7>

<item 8>
Anton Schwarzkopf
Anton Schwarzkopf (8 July 1924 – 30 July 2001) was a German engineer of amusement rides, and founder of the Schwarzkopf Industries Company, which built numerous
amusement rides and large roller coasters for both amusement parks and traveling funfairs.
</item 8>

<item 9>
Mack Rides
Mack Rides GmbH & Co KG, also known simply as Mack Rides, is a German company that designs and constructs amusement rides.

 Mack Rides in headquartered in Waldkirch, Baden-Württemberg, Germany.

 Mack Rides is one of the world's oldest amusement industry suppliers and builds all kinds of amusement devices including several types of flat rides, dark
rides, log flumes, tow boat rides and roller coasters.

 The family that owns Mack Rides also owns Europa-Park.
</item 9>

<item 10>
The entries cover roller-coaster history, manufacturers, parks, and attractions. Ice slides preceded French wooden-track coasters; steel coasters are now
widespread. Mack Rides and Japan’s TOGO build amusement rides; Swiss Giovanola supplied Intamin before producing coasters such as Goliath and Titan. Pretzel’s
company built over 1,400 single-rail dark rides. Notable coasters include Bandit, Millennium Force, Disney’s Space Mountains, and White Cyclone. Other
</item 10>
=== END ITEM id=0cc7a1720416cbab ===

=== ITEM id=15d096eccc6a0d8b ===
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
Daniel Boone Native Gardens
Daniel Boone Native Gardens, located in Boone, North Carolina, United States, has a collection of North Carolina native plants in an informal landscaped design.

 The gardens are open daily from May to October.
</item 1>

<item 2>
Kaydean
Kaydean is a British-American record and TV producer, composer, DJ, record label executive and artist manager.

 Kaydean produces, composes and arranges music in the Dance, Electronica, Latin Pop, Contemporary R&B, and Electronica genres.

 Born in London, he has worked with many artists including Janet Jackson, Maxi Priest, Robyn, Man Parrish, Angel "Cucco" Peña, Baron Lopez, Raekwon, Ky-Mani
Marley, Frankie Cutlass and Nocera amongst others.

 He is also known for discovering Latin Pop Star Janid and managing her career.

 He produced her Spanish album "La Magia" under the Sony Music Latin imprint label, Handshake and her Hit Single "Penicilina".

 In 2015 he debuted as a TV executive producer with Janid's reality show on Mega TV, The first episode aired on September 6, 2015.
</item 2>

<item 3>
Raleigh Moncrief
Raleigh Moncrief is an American electronic musician, record producer and recording engineer who has worked with Zach Hill, Dirty Projectors and !!!

.
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
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 5>

<item 6>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 6>

<item 7>
Anna Graceman
Anna Graceman is a young American singer-songwriter and pianist whose self-penned songs and music have gained attention at home and abroad.

 Her style of music incorporates elements of folk, soul and rock.

 As of 2016 she had written and performed over 70 of her own published songs; writing as well as producing her releases.

 Anna continues to divide her time between her solo projects and her new project - the ‘Graceman’ Band - a trio that includes Allie Graceman (Rhythm/Lead
Guitar) and Landon Graceman (Drums).
</item 7>

<item 8>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 8>

<item 9>
Zach Crowell
Zach Crowell is a Grammy Award-winning songwriter and record producer, who was born and raised in Nashville, Tennessee.

 Crowell is involved in a joint publishing deal with Warner/Chappell Music, Combustion Music and Taperoom Music.
</item 9>

<item 10>
J. Cole
Jermaine Lamarr Cole (born January 28, 1985), better known by his stage name J. Cole, is an American hip hop recording artist and record producer.

 Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.

 Intent on further pursuing a solo career as a rapper, he went on to release two additional mixtapes after signing to Jay Z's Roc Nation imprint in 2009.
</item 10>
=== END ITEM id=15d096eccc6a0d8b ===

=== ITEM id=2d048ec515ecb496 ===
Gold sentences and facts:
Sentence 0: Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
  Fact 0.0: Distortion Mirrors is an English alternative rock band.
  Fact 0.1: Distortion Mirrors is based in Leeds, West Yorkshire, England.
  Fact 0.2: Distortion Mirrors was formed in 2011.
Sentence 1: OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.
  Fact 1.0: OK Go is an American rock band.
  Fact 1.1: OK Go was originally from Chicago, Illinois.
  Fact 1.2: OK Go is now based in Los Angeles, California.

Retrieved items:
<item 1>
OK Go videography
The success of OK Go's music first won the band the 14th Annual Webby Special Achievement Award for Film and Video Artist of the Year.

 The video for "This Too Shall Pass" was named both "Video of the Year" and "Best Rock Video" at the 3rd annual UK Music Video Awards.

"This Too Shall Pass" won the LA Film Fest's Audience Award for Best Music Video, UK MVA Awards – Music Video of the Year Winner 2010, among others.
</item 1>

<item 2>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 2>

<item 3>
OK Go videography
The musical rock band OK Go has earned considerable fame for their creative but often low-budget music videos, most of which have been promoted through Internet
video sharing sites like YouTube.

 Many of these have become viral videos; the 2006 video for "Here It Goes Again", in which the band performed a complex routine with the aid of motorized
treadmills, has received over 50 million views four years later.

 Their video for Needing/Getting, released February 5, 2012 in partnership with Chevrolet, debuted during Super Bowl XLVI and has over 32 million views on
YouTube.

 Samuel Bayer, who produced many music videos in the 1990s, asserted that OK Go's promotion of music videos on the Internet was akin to Nirvana's ushering in
the grunge movement.

 Many of the videos also use long or single-shot takes, which "Salon"' s Matt Zoller Seitz says "restore[s] a sense of wonder to the musical number by letting
the performers' humanity shine through and allowing them to do their thing with a minimum of filmmaking interference".
</item 3>

<item 4>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 4>

<item 5>
OK Go (album)
OK Go is the debut studio album by American rock band OK Go.

 It was released in September 2002.

 The cover was created by designer Stefan Sagmeister.
</item 5>

<item 6>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 6>

<item 7>
Graph query row:
entity: OK Go
labels: ['__KGBuilder__', '__Entity__', 'Organization']
kind: band
genre: rock
relationship: LOCATED_IN
related_entity: Chicago
related_kind: city
related_genre: None
</item 7>

<item 8>
The Writing's on the Wall (OK Go song)
"The Writing's on the Wall" is a song by American rock band OK Go.

 It was released on June 17, 2014, as part of the band's EP "Upside Out", and is also the first single from the band's fourth studio album "Hungry Ghosts".

 On the same day, the band released a music video in which the members use props to create optical illusions, reflecting the song's description of a
relationship that fails because the couple has different points of view.

 Like previous OK Go videos, it is structured as a one-shot music video.

 The many YouTube views of the video caused the song to debut in the top ten of the US "Billboard" Hot Rock Songs chart, as well as number one on the Bubbling
Under Hot 100 Singles chart.
</item 8>

<item 9>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 9>

<item 10>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 10>
=== END ITEM id=2d048ec515ecb496 ===

=== ITEM id=a5d3d1823a83cb21 ===
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
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 1>

<item 2>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 2>

<item 3>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 3>

<item 4>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 4>

<item 5>
Power canal
A Power Canal refers to a canal used for hydraulic power generation, rather than for transport of watercraft.

 The power canal was a major factor in the Industrial revolution in New England in the 19th century.

 Most early power canals were mill races used mechanically to transfer power directly from falling water to machinery in mill buildings.

 Later, the hydraulic power generated electricity locally for the same mill factories.

 These power canals were often filled in as electricity (transported by power lines) replaced the need for local water power, and road transport needs or city
expansion needs reclaimed the land.

 Some hydraulic power canals were transformed into local electric generators, but most were closed.

 Remains of power canals can be seen in old mill towns and are often protected as historical structures today.
</item 5>

<item 6>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 6>

<item 7>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 7>

<item 8>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 8>

<item 9>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 9>

<item 10>
Edison Sault Power Canal
The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie,
Michigan.

 Excavation of the power canal began in September 1898 and was completed in June 1902.

 The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
</item 10>
=== END ITEM id=a5d3d1823a83cb21 ===

=== ITEM id=55a4123136dd10a0 ===
Gold sentences and facts:
Sentence 0: Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.
  Fact 0.0: Triumph of the Will's German title is Triumph des Willens.
  Fact 0.1: Triumph of the Will is a 1935 German propaganda film.
  Fact 0.2: Leni Riefenstahl directed Triumph of the Will.
  Fact 0.3: Leni Riefenstahl produced Triumph of the Will.
  Fact 0.4: Leni Riefenstahl edited Triumph of the Will.
  Fact 0.5: Leni Riefenstahl co-wrote Triumph of the Will.
Sentence 1: The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".
  Fact 1.0: The End of America is a 2008 documentary film.
  Fact 1.1: Annie Sundberg directed The End of America.
  Fact 1.2: Ricki Stern directed The End of America.
  Fact 1.3: The End of America is adapted from Naomi Wolf's 2007 book.

Retrieved items:
<item 1>
A Successful Failure
A Successful Failure (1934) is an American film directed by Arthur Lubin.

 It was Lubin's first movie as director.
</item 1>

<item 2>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 2>

<item 3>
Apocalypse Now
Apocalypse Now is a 1979 American epic war film directed, produced, and co-written by Francis Ford Coppola.

 It was co-written by John Milius with narration written by Michael Herr.

 It stars Marlon Brando, Robert Duvall, Martin Sheen, Frederic Forrest, Albert Hall, Sam Bottoms, Larry Fishburne, and Dennis Hopper.

 The screenplay, written by Milius, adapts the story of Joseph Conrad's novella "Heart of Darkness", changing its setting from late nineteenth-century Congo to
the Vietnam War.

 It draws from Herr's "Dispatches" and Werner Herzog's "Aguirre, the Wrath of God" (1972).

 The film revolves around Captain Benjamin L. Willard (Sheen) on a secret mission to assassinate Colonel Kurtz, a renegade who is presumed insane.
</item 3>

<item 4>
Triumph of the Will
Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.

 It chronicles the 1934 Nazi Party Congress in Nuremberg, which was attended by more than 700,000 Nazi supporters.

 The film contains excerpts from speeches given by Nazi leaders at the Congress, including Adolf Hitler, Rudolf Hess and Julius Streicher, interspersed with
footage of massed Sturmabteilung (SA) and Schutzstaffel (SS) troops and public reaction.

 Hitler commissioned the film and served as an unofficial executive producer; his name appears in the opening titles.

 The film's overriding theme is the return of Germany as a great power, with Hitler as the leader who will bring glory to the nation.

 Because the film was made after the 1934 Night of the Long Knives (on 30 June) many prominent Sturmabteilung (SA) members are absent—they were murdered in that
Party purge, organised and orchestrated by Hitler to replace the SA with the Schutzstaffel (SS) as his main paramilitary force.
</item 4>

<item 5>
Olympia (1938 film)
Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium
in Berlin, Germany.

 The film was released in two parts: "Olympia 1.

 Teil — Fest der Völker" (Festival of Nations) and "Olympia 2.

 Teil — Fest der Schönheit" (Festival of Beauty).

 It was the first documentary feature film of the Olympic Games ever made.

 Many advanced motion picture techniques, which later became industry standards but which were groundbreaking at the time, were employed —including unusual
camera angles, smash cuts, extreme close-ups, placing tracking shot rails within the bleachers, and the like.

 The techniques employed are almost universally admired, but the film is controversial due to its political context.

 Nevertheless, the film appears on many lists of the greatest films of all time, including "Time" magazine's "All-Time 100 Movies."
</item 5>

<item 6>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 6>

<item 7>
The End of America (film)
The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".

 The film analyzes policy changes made during the Bush Administration, and makes the case that these changes threaten American democracy.
</item 7>

<item 8>
Love Begins at 20
Love Begins at 20 is a 1936 American comedy film directed by Frank McDonald and written by Dalton Trumbo and Tom Reed, based on the 1929 play "Broken Dishes" by
Martin Flavin.

 The film stars Hugh Herbert, Patricia Ellis, Warren Hull, Hobart Cavanaugh, Dorothy Vaughan and Clarence Wilson.

 The film was released by Warner Bros. on August 22, 1936.
</item 8>

<item 9>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 9>

<item 10>
Empire (1964 film)
Empire is a 1964 American black and white silent film written, produced, and directed by Andy Warhol.

 It consists of eight hours and five minutes of continuous slow motion footage of the Empire State Building in New York City.

 Abridged showings of the film were never allowed, and supposedly the unwatchability of the film was an important part of the reason the film was created.

 However, a legitimate Italian VHS produced in association with the Andy Warhol Museum in 2000 contains only an extract of 60 minutes.

 Its use of the long take "in extremis" is an extension of Warhol's earlier work the previous year with "Sleep".

 Warhol employed Rob Trains to be the projectionist for a screening of the film.

 Trains miscalculated and mixed the order and speed of the reels for the eight-hour movie.

 After a positive review in "The New York Times", Warhol actually liked the "mistake" and employed Trains for the entire summer.
</item 10>
=== END ITEM id=55a4123136dd10a0 ===

=== ITEM id=d4c5cf5f1939267e ===
Gold sentences and facts:
Sentence 0:  The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
  Fact 0.0: The area along Northern Boulevard is well known for its high-end premium open-air shopping center.
  Fact 0.1: The high-end premium open-air shopping center is named the Americana Manhasset.
Sentence 1: International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine, Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van Sweden.
  Fact 1.0: Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine, Tiffany & Co., and Prada are international fashion designer labels and luxury goods companies.
  Fact 1.1: These companies maintain a total of more than 60 retail stores and restaurants.
  Fact 1.2: These stores and restaurants are on a 220000 sqft site.
  Fact 1.3: The 220000 sqft site is surrounded by gardens.
  Fact 1.4: The gardens surrounding the site were designed by Oheme van Sweden.

Retrieved items:
<item 1>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 1>

<item 2>
Americana Manhasset
International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine,
Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van
Sweden.

 Peter Marino is Americana Manhasset's master architect.
</item 2>

<item 3>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 3>

<item 4>
New Roc City
New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in
Westchester County, New York.

 It is located at 33 LeCount Place, between Main Street North and Main Street South.

 The center was built on the site of the former New Rochelle Mall which closed in 1992.
</item 4>

<item 5>
Luxify
Luxify is a Hong Kong-based company, providing an online marketplace for new, vintage and pre-owned luxury goods.

 The company manages an online marketplace website in which members buy and sell a variety of luxury goods.

 It was Hong Kong's first online marketplace to buy and sell luxury goods The site has over 15 categories, including real estate, watches, jewelry, luxury cars,
yachts, aircraft, luxury handbags, art, collectibles, furniture, antiques and fine wines & spirits.

 As of March 2016, the website had over 22,000 listings from luxury dealers from all over the world, making a US$18.6 billion marketplace.

 Luxify also has offices in Singapore and London and local partners in Shanghai and Jakarta.
</item 5>

<item 6>
Christian Dior SE
Christian Dior SE (] ), commonly known as Dior, is a French luxury goods company controlled and chaired by French businessman Bernard Arnault, who also heads
LVMH – the world's largest luxury group.

 Dior itself holds 42.36% shares of and 59.01% voting rights within LVMH.
</item 6>

<item 7>
Sunset Strip
Sunset Strip is the mile-and-a-half (2.4 km) stretch of Sunset Boulevard that passes through West Hollywood, California.

 It extends from West Hollywood's eastern border with Hollywood at Crescent Heights Boulevard to its western border with Beverly Hills at Sierra Drive.

 Sunset Strip is probably the best-known portion of Sunset Boulevard, with boutiques, restaurants, rock clubs, and nightclubs.

 It is also known for its array of huge, colorful billboards.
</item 7>

<item 8>
Miracle Mile (Manhasset)
The Miracle Mile is a prominent shopping district in Manhasset on the North Shore of Long Island in Nassau County, New York.

 The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
</item 8>

<item 9>
Mike Long (American businessman)
Mike Long is an American business man, former CEO of several public companies, and currently a founding partner of Sulgrave Partners LLC.

 He served as the President and Chief Executive Officer of Continuum, an Austin, Texas IT consulting company, from 1991 to 1997, having started with Continuum
as a Director in 1983.

 In 1997, Long was named CEO of Healtheon Corporation (now WebMD), succeeding former CEO David Schnell.

 Long oversaw Healtheon's initial public offering, traveling between Europe and the United States to woo investors.

 Long was able to secure the required investment funds, and saw Healtheon's stock price rise from $8 to a high of $120.

 In 2002, Long was recruited to fix the financial struggles of Move, Inc., a company plagued by more than $4 billion in lawsuits and hemorrhaging tens of
millions of dollars a quarter.

 As Chief Executive Officer, Long was able to revive Homestore, Inc., by changing the business model, rebranding the company as Move, Inc. and returning it to
profitability.
</item 9>

<item 10>
Mouawad
Mouawad is a privately held Swiss and Emirati luxury goods company that makes fashion accessories, jewelry and retail company that sources diamonds and
gemstones, designs, manufacturers, and sells jewelry collections, objects of art, and luxury watches.

 The firm has headquarters in Geneva, Switzerland, with a Middle East headquarters at Jumeirah Lakes Towers in Dubai.

 Founded in 1891 in Beirut, Lebanon by David Mouawad, the firm is now led by 4th generation co-guardians Fred Mouawad, Alain Mouawad and Pascal Mouawad.

 The firm's jewelry and watch design and manufacturing divisions are located in Switzerland and Asia Pacific.
</item 10>
=== END ITEM id=d4c5cf5f1939267e ===

=== ITEM id=225e1e1f44357749 ===
Gold sentences and facts:
Sentence 0: Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.
  Fact 0.0: Ash Lieb was born on 22 August 1982.
  Fact 0.1: Ash Lieb is an Australian artist.
  Fact 0.2: Ash Lieb is an Australian writer.
  Fact 0.3: Ash Lieb is an Australian comedian.
  Fact 0.4: Ash Lieb is known for his surreal humour.
  Fact 0.5: Ash Lieb is known for his surreal art.
Sentence 1: Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.
  Fact 1.0: Helen Elswyth Thane Ricker Beebe was born on May 16, 1900.
  Fact 1.1: Helen Elswyth Thane Ricker Beebe died on July 31, 1984.
  Fact 1.2: Helen Elswyth Thane Ricker Beebe was an American romance novelist.

Retrieved items:
<item 1>
Vladimir Ashkenazy
Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized
solo pianist, chamber music performer, and conductor.

 He is originally from Russia and has held Icelandic citizenship since 1972.

 He has lived in Switzerland since 1978.

 Ashkenazy has collaborated with well-known orchestras and soloists.

 In addition, he has recorded a large storehouse of classical and romantic works.

 His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
</item 1>

<item 2>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 2>

<item 3>
Hans Aschenborn
Hans Anton Aschenborn (1 February 1888 – 10 April 1931) was a renowned animal painter of African wildlife.

 He worked both in Germany and in southern Africa.

 His work is featured in the older German Thieme-Becker or Saur art encyclopedia.

 A Master of Arts thesis by Karin Skawran concerning the graphic works of Hans Anton Aschenborn was published in the South African art and culture periodical,
"Lantern" in 1965.)

 In 1963 the University of Pretoria published a book about Aschenborn as an artist entitled, "Hans Anton Aschenborn : Mens en Kunstenaar".

 In 1970 another one followed by the Pretoria Art Museum (South Africa).

 Other publications on Aschenborn feature his etchings and linocuts.
</item 3>

<item 4>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 4>

<item 5>
Ash Lieb
Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.

 Born in Ballarat, Ash Lieb began exhibiting art at eight years of age, and at the age of fifteen, wrote his first novel, "The Secret Well".

 Throughout his career, Lieb has created a diverse range of artworks, books, short films, and comedic performances, which have often possessed philosophical or
psychiatric undertones.
</item 5>

<item 6>
Elswyth Thane
Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.

 Born in Burlington, Iowa, she was the daughter of a local teacher and high school principal.

 The family moved to New York City in 1918, and "Helen Ricker" changed her name to "Elswyth Thane".

 She began working as a freelance writer in the 20s, and became a newspaper writer and a Hollywood screenwriter.

 Her first novel, "Riders of the Wind", was published in 1926.

 Her novel, "The Tudor Wench", about Elizabeth I of England, was made into a play.

 She was a collector of scarves.
</item 6>

<item 7>
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
</item 7>

<item 8>
Tryst (novel)
Tryst, written in 1939 by Elswyth Thane, is a story of two people and a seemingly ordinary home.

 While a quick summary may make it sound like a Horror novel, it actually borders on Mystery and Romance.
</item 8>

<item 9>
Oliver Lieb
Oliver Lieb (born 1969 in Frankfurt, Germany) is a German electronic music producer and DJ.

 Lieb is known to have more than a dozen aliases with over 200 productions and remixes in various electronic genres such as trance, house, and techno.

 He has also produced tracks for other artists.
</item 9>

<item 10>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 10>
=== END ITEM id=225e1e1f44357749 ===

=== ITEM id=0c11f6d390de9395 ===
Gold sentences and facts:
Sentence 0: "Grow Some Funk of Your Own" is a song by Elton John.
  Fact 0.0: 'Grow Some Funk of Your Own' is a song by Elton John.
Sentence 1: Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.
  Fact 1.0: Sir Elton Hercules John was born Reginald Kenneth Dwight.
  Fact 1.1: Sir Elton Hercules John was born on 25 March 1947.
  Fact 1.2: Sir Elton Hercules John is an English singer.
  Fact 1.3: Sir Elton Hercules John is an English pianist.
  Fact 1.4: Sir Elton Hercules John is an English composer.
Sentence 2:  John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.
  Fact 2.0: Elton John owned Watford Football Club from 1976 to 1987.
  Fact 2.1: Elton John owned Watford Football Club from 1997 to 2002.

Retrieved items:
<item 1>
Firm Biz
"Firm Biz" is the first single by the Hip hop supergroup The Firm from their singular collaborative LP "The Album".

 The song was produced by L.E.S, who based the song's track on a sample of Teena Marie's 1981 hit "Square Biz".

 "Firm Biz" also features a chorus performed by former En Vogue singer Dawn Robinson, reworked from the "Square Biz" chorus.

 AZ raps the first verse of the song, Nas the second, and Foxy Brown the third and final verse.

 The song never reached any US chart positions, but was a UK Top 20 hit in 1997.

 A remix of the song was released, featuring a verse from Half-A-Mill and a chorus by Mary J. Blige.

 The word "feds" is censored in the explicit version, in verse 2, performed by Nas, when he says "Never that though Black .4-4's for feds".
</item 1>

<item 2>
Afrodiziak
Afrodiziak was a British singing group composed of Caron Wheeler, Claudia Fontaine, and later Naomi Thompson, that was active in the 1980s.

 As a duo, Wheeler and Fontaine were best known for performing backing vocals on The Jam's final single "Beat Surrender" in 1982 (with whom they also performed
on their final tour) and Elvis Costello's 1983 album "Punch the Clock", especially its single, the international hit "Everyday I Write the Book".

 After the addition of a third member, Naomi Thompson, they went on to sing for acts such as Heaven 17 and Howard Jones.

 Afrodiziak performed backing vocals on the hit single Free Nelson Mandela, including the a cappella intro, staging it at The Tube (Channel4) in March 1984.

 Heaven 17's 1984 album, "How Men Are", featured them prominently, especially on the singles "Sunset Now" and "And That's No Lie".
</item 2>

<item 3>
Elton John
He has also composed music, produced records, and has occasionally acted in films.

 John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.

 He is an honorary Life President of the club, and in 2014 had a stand named after him at the club's home stadium.
</item 3>

<item 4>
Grow Some Funk of Your Own
"Grow Some Funk of Your Own" is a song by Elton John.

 It was released as a single in 1976 and comes from his album "Rock of the Westies".

 It shared its A-side status with "I Feel Like a Bullet (In the Gun of Robert Ford)".

 The song is considered to be one of John's heavier rock songs.

 The song went to #14 on the "Billboard" Hot 100, but in Britain broke a five-year run of successful singles by failing to reach the Top 50 despite extensive
radio play.

 Although the singles chart listed only "Funk" and not "Bullet", the latter song was given equal billing in later "Billboard" singles chart books.
</item 4>

<item 5>
Knitting Factory Records
Knitting Factory Records is an independent American music label that is notable for promoting a variety of artists, including the music of deceased Nigerian
political activist Fela Kuti.

 The label promotes a variety of music artists including Ages and Ages, Ash Black Bufflo, Cuong Vu, Graham Haynes, Femi Kuti, Gary Lucas, Lumerians, Thomas
Chapin, Patrolled By Radar, Joe Morris, Rachid Taha, Seun Kuti, and Shilpa Ray and her Happy Hookers.
</item 5>

<item 6>
Blitz Kids
The Blitz Kids were a group of young people who frequented the weekly Blitz club-night in Covent Garden, London in 1979-80, and are credited with launching the
New Romantic subcultural movement.

 It was Andrew Czezowski (Ex manager of The Damned) and Susan Carrington who went on to start The Roxy club with Barry Jones that introduced Steve Strange and
Rusty Egan to the Blitz in Covent Garden, which started the New Romantic Movement.

 Among the many attendees were Rusty Egan, Steve Strange, Boy George, Marilyn, Alice Temple, Perri Lister, Princess Julia, Philip Sallon, Carl Teper and Martin
Degville (later to be the frontman of Tony James' Sigue Sigue Sputnik).

 Crucially, the club lay between two art colleges (St Martin's School and Central School) and became a testbed for student fashion designers who set London
ablaze during the 1980s.
</item 6>

<item 7>
A Boy in a Man's World
A Boy in a Man's World is the second studio album by Mucky Pup.

 The album was recorded at Nevessa Productions and released in 1989 through Torrid Records and Roadrunner Records.

 The album features a re-recorded version of "U-Stink-But-I-♥-U," which the band had previously won a Bloom County songwriting contest with.

 The band would also shoot their first music video for this song.

 The album also features "Batman", a song based on an urban legend that also contains a snippet of the title theme to the 1966 "Batman" TV.
</item 7>

<item 8>
South Liverpool F.C.
South Liverpool Football Club is a football club based in Liverpool, England, founded as a phoenix club of a club of the same name.

 They are currently members of the West Cheshire League Division One and play at Jericho Lane in Otterspool.

 The club's colours are white shirts, black shorts and red socks.
</item 8>

<item 9>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 9>

<item 10>
The Happenings
The Happenings are a pop music group that originated in the 1960s.

 Members of the original group, created in the spring of 1961 and initially called "The Four Grads" because all had just graduated from high school in Paterson,
New Jersey, were Bob Miranda, David Libert, Tom Giuliano, and Ralph DiVito.

 In 1968 DiVito was replaced by Bernie LaPorta and Lenny Conforti also joined to play drums in the touring band.

 Both LaPorta and Conforti took a hiatus from the northern New Jersey band, The Emerald Experience, to play and tour with The Happenings.

 That lineup performed mostly at colleges and universities until 1970, when Libert left the band to manage other groups, including George Clinton and
Parliament-Funkadelic, Living Colour, Brian Auger, Vanilla Fudge, The Runaways (Cherie Currie, Joan Jett, Lita Ford), Mother's Finest, Alice Cooper and Evelyn
"Champagne" King.
</item 10>
=== END ITEM id=0c11f6d390de9395 ===

=== ITEM id=b93ab64120c15112 ===
Gold sentences and facts:
Sentence 0:  The album was Prince's third to be released that year (along with "Chaos and Disorder" and the soundtrack album of the Spike Lee movie "Girl 6"), which made 1996 one of the most prolific years for material released by Prince.
  Fact 0.0: The album was Prince's third album released in 1996.
  Fact 0.1: One of Prince's other 1996 releases was "Chaos and Disorder".
  Fact 0.2: One of Prince's other 1996 releases was the soundtrack album of the Spike Lee movie "Girl 6".
  Fact 0.3: 1996 was one of the most prolific years for material released by Prince.
Sentence 1:  His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
  Fact 1.0: Prince's production company is 40 Acres and a Mule Filmworks.
  Fact 1.1: 40 Acres and a Mule Filmworks has produced over 35 films since 1983.

Retrieved items:
<item 1>
The entries span film, television, documentaries, and film distribution: Philippine series Sine Novela adapts Viva classics; The Doom Generation (1995) is Gregg
Araki’s violent “Teenage Apocalypse” film; Naked Gun 33⅓ concludes its comedy trilogy, while Naked City (1958–63) is a semi-documentary police drama. Night Life
is a zombie comedy; The Runaways portrays the band’s formation and Currie-Jett relationship. Consolidated Film Industries was
</item 1>

<item 2>
Ian Schneider is an independent American director and screenwriter. Mary Sweeney, a longtime David Lynch collaborator, edited and produced *Twin Peaks*, *Lost
Highway*, *The Straight Story* and *Mulholland Drive*; she teaches at USC and chairs Film Independent. Angelo Badalamenti scored Lynch works and won a Grammy.
Brian Ralston, Nancy Schreiber and John Bailey are established composers/cinematographers; Bailey led the Motion Picture Academy. Jon Paul Puno is
</item 2>

<item 3>
Six Cylinder Love
Six Cylinder Love is a lost 1923 silent film comedy produced and distributed by Fox Film and directed by Elmer Clifton.

 It is based on a popular 1921 Broadway play and stars Ernest Truex from the play.

 Also appearing in the film from the Broadway play were Donald Meek and Ralph Sipperly.
</item 3>

<item 4>
Cannibal boom
Only four other cannibal films were made after 1981 until the fad's conclusion in 1988 with Antonio Climati's "Natura Contro".
</item 4>

<item 5>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 5>

<item 6>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 6>

<item 7>
*Girl 6* (1996), Spike Lee’s comedy-drama about a phone-sex operator, stars Theresa Randle and was written by Suzan-Lori Parks. It was Lee’s first film without
his own screenplay, features an all-Prince soundtrack, and screened at Cannes; Quentin Tarantino and Ron Silver cameo. Other profiles cover Issa Rae, creator of
*Awkward Black Girl* and a platform for creators of color; *Family Matters* actress Jaimee Foxworth;
</item 7>

<item 8>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 8>

<item 9>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 9>

<item 10>
*Girl 6* is Spike Lee’s 1996 comedy-drama about a phone sex operator, starring Theresa Randle and written by playwright Suzan-Lori Parks. It was Lee’s first
film without a screenplay by him; its soundtrack consists entirely of Prince songs. Screened at Cannes in Un Certain Regard, it features director cameos by
Quentin Tarantino and Ron Silver. Randle’s other films include *Malcolm X*, *Bad Boys*, *Space Jam*, and *Spawn
</item 10>
=== END ITEM id=b93ab64120c15112 ===

=== ITEM id=f5297fc85d60af9c ===
Gold sentences and facts:
Sentence 0: Sir Ridley Scott (born 30 November 1937) is an English film director and producer.
  Fact 0.0: Sir Ridley Scott was born on 30 November 1937.
  Fact 0.1: Sir Ridley Scott is English.
  Fact 0.2: Sir Ridley Scott is a film director.
  Fact 0.3: Sir Ridley Scott is a film producer.
Sentence 1:  Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
  Fact 1.0: Ridley Scott had a commercial breakthrough with the science-fiction horror film Alien, released in 1979.
  Fact 1.1: Ridley Scott's best known works include the neo-noir dystopian science fiction film Blade Runner, released in 1982.
  Fact 1.2: Ridley Scott's best known works include the historical drama Gladiator, released in 2000.
  Fact 1.3: Gladiator won the Best Picture Oscar.
  Fact 1.4: Ridley Scott's best known works include the science fiction film The Martian, released in 2015.
Sentence 2:  A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work behind the camera, with Griffith and Joseph Henabery as his mentors.
  Fact 2.0: Ridley Scott was a collaborator of D.W. Griffith.
  Fact 2.1: Ridley Scott appeared in The Birth of a Nation, released in 1915.
  Fact 2.2: Ridley Scott appeared in Intolerance, released in 1916.
  Fact 2.3: Ridley Scott gave up acting in 1917.
  Fact 2.4: Ridley Scott gave up acting to concentrate on work behind the camera.
  Fact 2.5: Griffith was a mentor of Ridley Scott.
  Fact 2.6: Joseph Henabery was a mentor of Ridley Scott.

Retrieved items:
<item 1>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 1>

<item 2>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 2>

<item 3>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 3>

<item 4>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 4>

<item 5>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 5>

<item 6>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 6>

<item 7>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 7>

<item 8>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 8>

<item 9>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 9>

<item 10>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 10>
=== END ITEM id=f5297fc85d60af9c ===

=== ITEM id=69c1c289a162b8c0 ===
Gold sentences and facts:
Sentence 0: Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public school students under the First Amendment to the United States Constitution.
  Fact 0.0: Minersville School District v. Gobitis, 310 U.S. 586 (1940), was a decision by the Supreme Court of the United States.
  Fact 0.1: Minersville School District v. Gobitis involved the religious rights of public school students.
  Fact 0.2: Minersville School District v. Gobitis involved religious rights under the First Amendment to the United States Constitution.
Sentence 1: Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.
  Fact 1.0: Gravel v. United States, 408 U.S. 606 (1972), was a case.
  Fact 1.1: Gravel v. United States concerned the protections offered by the Speech or Debate Clause of the United States Constitution.

Retrieved items:
<item 1>
Forest Grove School District v. T. A.
Forest Grove School District v. T. A., 557 U.S. 230 (2009), is a case in which the United States Supreme Court held that the Individuals with Disabilities
Education Act (IDEA) authorizes reimbursement for private special education services when a public school fails to provide a "free appropriate public education"
(FAPE) and the private school placement is appropriate, regardless of whether the child previously received special education services through the public
school.
</item 1>

<item 2>
Minersville School District v. Gobitis
Minersville School District v. Gobitis, 310 U.S. 586 (1940) , was a decision by the Supreme Court of the United States involving the religious rights of public
school students under the First Amendment to the United States Constitution.

 The Court ruled that public schools could compel students—in this case, Jehovah's Witnesses—to salute the American Flag and recite the Pledge of Allegiance
despite the students' religious objections to these practices.

 This decision led to increased persecution of Witnesses in the United States.

 The Supreme Court overruled this decision a mere three years later, in "West Virginia State Board of Education v. Barnette", 319 U.S. 624 (1943).
</item 2>

<item 3>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 3>

<item 4>
Cedar Rapids Community School Dist. v. Garret F.
Cedar Rapids Community School District v. Garret F., 526 U.S. 66 (1999) , was a United States Supreme Court case in which the Court ruled that the related
services provision in the Individuals with Disabilities Education Act (IDEA) required public school districts to fund "continuous, one-on-one nursing care for
disabled children" despite arguments from the school district concerning the costs of the services."

 The judges relied heavily on Irving Independent School Dist.

 v. Tatro, 468 U. S. 883 (1984).

 Under the Court's reading of the IDEA's relevant provisions, medical treatments such as suctioning, ventilator checks, catheterization, and others which can be
administered by non-physician personnel come within the parameters of the special education law's related services.

 Disability advocates considered the Court decision to be a "substantial victory for families of children with disabilities."

 Amendments were made in the Education Flexibility Partnership Act of 1999 to increase IDEA funding as a result of the case.
</item 4>

<item 5>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 5>

<item 6>
Bethel School District v. Fraser
Bethel School District v. Fraser, 478 U.S. 675 (1986), was a landmark decision by the United States Supreme Court involving free speech in public schools.

 High school student Matthew Fraser was suspended from school in the Bethel School District for making a speech including sexual double entendres at a school
assembly.

 The Supreme Court held that his suspension did not violate the First Amendment.
</item 6>

<item 7>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 7>

<item 8>
Gravel v. United States
Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.

 In the case, the Supreme Court of the United States held that the privileges and immunities of the Constitution's Speech or Debate Clause enjoyed by members of
Congress also extend to Congressional aides, but not to activity outside the legislative process.
</item 8>

<item 9>
Safford Unified School District v. Redding
Safford Unified School District v. Redding, 557 U.S. 364 (2009) , was a United States Supreme Court case in which the Court held that a strip search of a middle
schooler violated the Fourth Amendment to the United States Constitution where the school lacked reasons to suspect either that the drugs (Ibuprofen) presented
a danger or that they were concealed in her underwear.

 The court also held, however, that because this was not clearly established law prior to the court's decision, the officials involved were shielded from
liability by qualified immunity.
</item 9>

<item 10>
Elk Grove Unified School District v. Newdow
Elk Grove Unified School District v. Newdow, 542 U.S. 1 (2004), was a case decided by the U.S. Supreme Court.

 The lawsuit, originally filed as Newdow v. United States Congress, Elk Grove Unified School District, et al. in 2000, led to a 2002 ruling by the United States
Court of Appeals for the Ninth Circuit that the words "under God" in the Pledge of Allegiance are an endorsement of religion and therefore violate the
Establishment Clause of the First Amendment to the United States Constitution.

 After an initial decision striking the congressionally added "one nation under God" language, [Newdow v. U.S. Congress, 292 F.3d 597 (9th Cir.

 2002), the superseding opinion on denial of rehearing en banc was more limited, holding that compelled recitation of the language by school teachers to
students was invalid.

 [328 F.3d 466 (9th Cir.

 2003).
</item 10>
=== END ITEM id=69c1c289a162b8c0 ===
