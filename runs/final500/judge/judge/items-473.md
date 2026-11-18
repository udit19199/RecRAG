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

This file has 10 items.

=== ITEM id=70a7f8d4a1330e68 ===
Gold sentences and facts:
Sentence 0: Mastodon is an American heavy metal band from Atlanta, Georgia.
  Fact 0.0: Mastodon is an American heavy metal band.
  Fact 0.1: Mastodon is from Atlanta, Georgia.
Sentence 1:  Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals in studio (except Bill Kelliher).
  Fact 1.0: Mastodon was formed in 2000.
  Fact 1.1: Mastodon is composed of bassist Troy Sanders.
  Fact 1.2: Mastodon is composed of guitarist Brent Hinds.
  Fact 1.3: Mastodon is composed of guitarist Bill Kelliher.
  Fact 1.4: Mastodon is composed of drummer Brann Dailor.
  Fact 1.5: Troy Sanders, Brent Hinds, and Brann Dailor perform vocals in the studio.
  Fact 1.6: Bill Kelliher does not perform vocals in the studio.
Sentence 2:  Mastodon has released seven studio albums, as well as a number of other records.
  Fact 2.0: Mastodon has released seven studio albums.
  Fact 2.1: Mastodon has released a number of other records.
Sentence 3:  Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.
  Fact 3.0: Mastodon's second full-length release is "Leviathan".
  Fact 3.1: "Leviathan" is a concept album.
  Fact 3.2: "Leviathan" is based on the novel "Moby-Dick".
  Fact 3.3: "Moby-Dick" was written by Herman Melville.
Sentence 4: Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric Erlandson.
  Fact 4.0: Hole was an American alternative rock band.
  Fact 4.1: Hole formed in Los Angeles, California in 1989.
  Fact 4.2: Hole was formed by singer and guitarist Courtney Love.
  Fact 4.3: Hole was formed by lead guitarist Eric Erlandson.

Retrieved items:
<item 1>
Hole (band)
Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric
Erlandson.

 The band had a revolving line-up of bassists and drummers, their most prolific being drummer Patty Schemel, and bassists Kristen Pfaff (d. 1994) and Melissa
Auf der Maur.
</item 1>

<item 2>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 2>

<item 3>
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 3>

<item 4>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 4>

<item 5>
Creed (band)
Creed is an American rock band formed in 1993 in Tallahassee, Florida.

 The band's best-known line-up consists of lead vocalist Scott Stapp, guitarist and vocalist Mark Tremonti, bassist Brian Marshall, and drummer Scott Phillips.

 Creed released two studio albums, "My Own Prison" in 1997 and "Human Clay" in 1999, before Marshall left the band in 2000.

 The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between
members.

 Tremonti, Marshall, and Phillips went on to found Alter Bridge while Stapp followed a solo career.
</item 5>

<item 6>
Mastodon (band)
Mastodon is an American heavy metal band from Atlanta, Georgia.

 Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals
in studio (except Bill Kelliher).

 They all perform vocals at live shows.

 Their musical style features progressive concepts and unique instrumentation.

 Mastodon has released seven studio albums, as well as a number of other records.

 The band's debut album, "Remission", released in 2002, garnered significant critical acclaim for its unique sound.

 Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.

 Three magazines awarded the record "Album of the Year" in 2004: "Revolver", "Kerrang!

" and "Terrorizer".
</item 6>

<item 7>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 7>

<item 8>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 8>

<item 9>
Blackdiamondskye
Blackdiamondskye was a 2010 North American concert tour by American bands Alice in Chains, Mastodon, and Deftones.

 Blackdiamondskye is a portmanteau of the three band's most recent albums: Alice in Chains' "Black Gives Way to Blue", Deftones' "Diamond Eyes", and Mastodon's
"Crack the Skye".

 The tour kicked off on September 16 in Chicago and culminated on October 16 in Paradise, Nevada.

 The tour lasted for 19 shows over the course of one month and featured shows at both outdoor amphitheatres and indoor venues.
</item 9>

<item 10>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 10>
=== END ITEM id=70a7f8d4a1330e68 ===

=== ITEM id=5f7629a4561b86e1 ===
Gold sentences and facts:
Sentence 0:  Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.
  Fact 0.0: Watkins was born in Des Moines, Iowa.
  Fact 0.1: Watkins rose to fame in the early 1990s.
  Fact 0.2: Watkins rose to fame as a member of the girl-group TLC.
Sentence 1:  She is best known as the former main vocalist and member of South Korean girl group T-ara.
  Fact 1.0: Watkins is best known as the former main vocalist of the South Korean girl group T-ara.
  Fact 1.1: Watkins is best known as the former member of the South Korean girl group T-ara.

Retrieved items:
<item 1>
Rain (Taeyeon song)
"Rain" is a song by South Korean singer Kim Tae-yeon, a member of the South Korean girl group Girls' Generation.

 It was released digitally by S.M. Entertainment on February 3, 2016.
</item 1>

<item 2>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 2>

<item 3>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 3>

<item 4>
Kim Tae-yeon (actress)
Kim Tae-yeon (born January 3, 1976) is a South Korean actress.

 She began her entertainment career as a model, winning Model Line's 40th Fashion Model contest in 1996 and the Pantene Model contest sponsored by Ford Models
in 2000.

 Kim made her film debut in the highly controversial film "Lies" in 1999.
</item 4>

<item 5>
TLC (group)
TLC is an American girl group whose original line-up consisted of Tionne "T-Boz" Watkins, Lisa "Left Eye" Lopes and Rozonda "Chilli" Thomas.

 Formed in Atlanta, Georgia in 1990, the group was very successful during the 1990s despite numerous spats with the law, each other, and the group's record
label and management.

 They scored nine top-ten hits on the "Billboard" Hot 100, including four number-one singles "Creep", "Waterfalls", "No Scrubs", and "Unpretty".

 The group also recorded four multi-platinum albums, including "CrazySexyCool" (1994) which still remains the only album by a female group to receive a diamond
certification from the Recording Industry Association of America (RIAA).

 TLC also became the first R&B group in history to receive Million certification from the Recording Industry Association of Japan (RIAJ) for "FanMail" (1999).
</item 5>

<item 6>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 6>

<item 7>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 7>

<item 8>
R U the Girl
R U the Girl (also known as R U the Girl with T-Boz and Chilli) is an American reality series that aired on UPN in 2005.

 The series featured Tionne "T-Boz" Watkins and Rozonda "Chilli" Thomas, the remaining members of the all-girl R&B group TLC whose former member, Lisa "Left
Eye" Lopes, died in a car crash in Honduras on April 25, 2002.

 Initially promoted as a contest to permanently replace Lopes 3 years after her death by TLC themselves, both Watkins and Thomas admitted that the winner of the
contest would not be joining TLC full-time and would not be a full-time replacement member; the winner would only provide guest vocals on a new single by the
duo.
</item 8>

<item 9>
Tionne Watkins
Tionne Tenese Watkins (born April 26, 1970), better known by her stage name T-Boz, is an American singer, songwriter, actress, author, and executive producer.

 Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.

 She has won four Grammy Awards for her work with TLC.
</item 9>

<item 10>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 10>
=== END ITEM id=5f7629a4561b86e1 ===

=== ITEM id=ae7e70677a99e2e1 ===
Gold sentences and facts:
Sentence 0: The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.
  Fact 0.0: The Long–Allen Bridge is a truss bridge.
  Fact 0.1: The Long–Allen Bridge is in Shreveport, Louisiana.
  Fact 0.2: The Long–Allen Bridge is named for Louisiana governor Huey Long.
  Fact 0.3: The Long–Allen Bridge is named for Louisiana governor Oscar K. Allen.
Sentence 1:  Opened in 1933, it carries U.S. Route 80 across the Red River.
  Fact 1.0: The Long–Allen Bridge opened in 1933.
  Fact 1.1: The Long–Allen Bridge carries U.S. Route 80 across the Red River.
Sentence 2: U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.
  Fact 2.0: U.S. Route 80 (US 80) is an east–west United States Numbered Highway.
  Fact 2.1: Much of U.S. Route 80 was once part of the early auto trail known as the Dixie Overland Highway.
Sentence 3:  The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
  Fact 3.0: The eastern terminus of U.S. Route 80 is in Tybee Island, Georgia.
  Fact 3.1: The eastern terminus of U.S. Route 80 is at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street.
  Fact 3.2: The eastern terminus of U.S. Route 80 is near the Atlantic Ocean.

Retrieved items:
<item 1>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 1>

<item 2>
Broad Street Bridge (Rochester, New York)
Erie Canal: Second Genesee Aqueduct, also known as the Broad Street Aqueduct or Broad Street Bridge, is a historic stone aqueduct located at Rochester in Monroe
County, New York.

 It was constructed in 1836-1842 and originally carried the Erie Canal over the Genesee River.

 The overall length of the aqueduct including the wings and abutments is 800 ft .

 The aqueduct is 70 ft wide and has massive parapets on either side.

 It is one of four major aqueducts in the mid-19th century Erie Canal system.

 In 1927, a roadbed was added to carry automobile traffic and named Broad Street.

 It also carried a part of the Rochester Subway.
</item 2>

<item 3>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 3>

<item 4>
U.S. Route 80
U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.

 As the "0" in the route number indicates, it was originally a cross-country route, from the Pacific Ocean to the Atlantic Ocean.

 However, the entire segment west of Dallas, Texas, has been decommissioned in favor of various Interstate Highways and state highways.

 Currently, the highway's western terminus is at an interchange with Interstate 30 (I-30) on the Dallas–Mesquite, Texas city line.

 The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
</item 4>

<item 5>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 5>

<item 6>
Long–Allen Bridge (Shreveport)
The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.

 Opened in 1933, it carries U.S. Route 80 across the Red River.

 It is also known as the Texas Street Bridge.
</item 6>

<item 7>
Newark Bay
Newark Bay is a tidal bay at the confluence of the Passaic and Hackensack Rivers in northeastern New Jersey.

 It is home to the Port Newark-Elizabeth Marine Terminal, the largest container shipping facility in Port of New York and New Jersey, the third largest and one
of the busiest in the United States.

 An estuary, it is periodically dredged to accommodate ocean-going ships.
</item 7>

<item 8>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 8>

<item 9>
Rocky Creek Bridge No. 01089
Rocky Creek Bridge No. 01089, also known as Ben Jones Bridge, is a concrete highway bridge spanning Rocky Creek along the Pacific Ocean coast of the U.S. state
of Oregon.

 The bridge crosses a gorge near the creek's mouth, about 2 mi south of Depoe Bay in Lincoln County.

 Built in 1927, the bridge originally carried U.S. Route 101 (the Oregon Coast Highway) over the creek.

 After a stretch of the highway was relocated to make it straighter, the abandoned piece of the old highway became Otter Crest Loop Road, a bypass west of the
new highway.
</item 9>

<item 10>
Grand Central Parkway
The Grand Central Parkway (GCP) is a 14.61-mile (23.51 km) long parkway that stretches from the Triborough Bridge in New York City to Nassau County on Long
Island.

 At the Queens–Nassau border, it becomes the Northern State Parkway, which runs across the northern part of Long Island through Nassau County and into Suffolk
County, where it ends in Hauppauge.

 The westernmost stretch (from the Triborough Bridge to exit 4) also carries a short stretch of Interstate 278 (I-278).

 The parkway runs through Queens and passes the Cross Island Parkway, Long Island Expressway, LaGuardia Airport and Citi Field, home of the New York Mets.

 The parkway is designated New York State Route 907M (NY 907M), an unsigned reference route.

 Despite its name, the Grand Central Parkway was not named after Grand Central Terminal.
</item 10>
=== END ITEM id=ae7e70677a99e2e1 ===

=== ITEM id=b218fe88c21f2a1a ===
Gold sentences and facts:
Sentence 0: Bank Street () is a short street in Central, Hong Kong.
  Fact 0.0: Bank Street is a short street.
  Fact 0.1: Bank Street is located in Central, Hong Kong.
Sentence 1:  It links Des Voeux Road Central to Queen's Road Central.
  Fact 1.0: Bank Street links Des Voeux Road Central to Queen's Road Central.
Sentence 2:  It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong) respectively.
  Fact 2.0: Bank Street is named after its location between the HSBC Building and the Bank of China Building.
  Fact 2.1: The HSBC Building was the former headquarters of HSBC.
  Fact 2.2: The Bank of China Building was the former headquarters of Bank of China (Hong Kong).
Sentence 3: HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC Holdings.
  Fact 3.0: HSBC Main Building is a headquarters building of The Hongkong and Shanghai Banking Corporation.
  Fact 3.1: The Hongkong and Shanghai Banking Corporation is today a wholly owned subsidiary of London-based HSBC Holdings.

Retrieved items:
<item 1>
HSBC Building, the Bund
The HSBC Building (also known as the Municipal Government Building) is a six-floor neo-classical building in the Bund area of Shanghai, China.

 It served as the headquarters of the Shanghai branch of The Hongkong and Shanghai Banking Corporation from 1923 to 1955, and currently houses the Shanghai
Pudong Development Bank.

 The building is situated at number 12, the Bund.

 Designed by the British architecture firm Palmer & Turner Architects and Surveyors, construction of the building lasted from 5 May 1921 to 23 June 1923.
</item 1>

<item 2>
Hong Kong fifty-dollar note
The fifty-dollar note was first issued undated in the 1860s by the Oriental Bank Corporation, the Mercantile Bank, the Standard Chartered Bank (Hong Kong) but a
confirmed date for this bank is 1879, followed by The Hongkong and Shanghai Banking Corporation in 1877.

 The National Bank of China also issued this denomination in the 1890s, but they are seldom seen.

 There was a continuous issue till the Second World War in different colours and dimensions, but they ceased to be printed between 1934–41, depending on the
bank.

 After the war no banks resumed to issue this denomination.

 They were resumed by The Hongkong and Shanghai Banking Corporation in 1968 the Standard Chartered Bank in 1970 as a blue note.

 This was then changed to purple in 1985 with a new smaller version and then to the current green issue in 2004.

 The Bank of China issued their version in 1994.

 The colour was made uniform when green for all banknotes was adopted.
</item 2>

<item 3>
Hong Kong twenty five-dollar note
The Hong Kong twenty five dollar note was first issued from 1864 by the Oriental Bank Corporation, The Hongkong and Shanghai Banking Corporation in 1865, the
Standard Chartered Bank (Hong Kong) from 1879 followed by the Mercantile Bank in 1889, though specimens of an earlier date exist.

 Specimens are known from the Asiatic Banking Corporation that existed between 1862-66.

 This denomination was last printed in 1912 by the Mercantile Bank.
</item 3>

<item 4>
HSBC
HSBC Holdings PLC is a British multinational banking and financial services holding company, tracing its origin to a hong in Hong Kong.

 It is the world's seventh largest bank by total assets and the largest in Europe with total assets of US$2.374 trillion (as of December 2016).

 It was established in its present form in London in 1991 by The Hongkong and Shanghai Banking Corporation Limited to act as a new group holding company.

 The origins of the bank lie mainly in Hong Kong and to a lesser extent in Shanghai, where branches were first opened in 1865.

 The HSBC name is derived from the initials of the Hongkong and Shanghai Banking Corporation.

 The company was first formally incorporated in 1866.

 The company continues to see both the United Kingdom and Hong Kong as its "home markets".
</item 4>

<item 5>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 5>

<item 6>
HSBC Building (Hong Kong)
HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC
Holdings.

 It is located along the southern side of Statue Square near the location of the old City Hall, Hong Kong (built in 1869, demolished in 1933).

 The previous HSBC building was built in 1935 and pulled down to make way for the current building.

 The address remains as 1 Queen's Road Central (the north facing side of the building was served by Des Voeux Road, which was seashore, making Queen's Road was
the main entrance, in contrast to modern day that usually accessed the bank from Des Voeux Road).

 The building can be reached from Exit K of Central MTR Station and facing Statue Square.
</item 6>

<item 7>
Hong Kong one hundred-dollar note
The Hong Kong one hundred dollar note was first issued from 1858 from the Mercantile Bank, 1866 by the Oriental Bank Corporation, the Standard Chartered Bank
(Hong Kong) from the 1860s but a confirmed date for this bank is 1879, followed by The Hongkong and Shanghai Banking Corporation in 1877.

 Specimens are known from the Agra and Masterman's Bank and the Asiatic Banking Corporation that existed between 1862–66 and from The National Bank of China in
the 1890s.

 There was a continuous issue till the Second World War in different colours and dimensions, and this issue was resumed after the war in 1946, by the HSBC,
Mercantile and Standard Chartered Banks.

 This was somewhat standardised in 1970 when the Chartered Bank changed the issue from brown to red, red was the colour of the other two issues.

 The Mercantile bank stopped issuing banknotes after 1974 and the Bank of China issued their version in 1994.

 The colour was made uniform when red for all banknotes was adopted.
</item 7>

<item 8>
Hong Kong five hundred-dollar note
The Hong Kong five hundred dollar note was first issued in undated from the 1860s by the Oriental Bank Corporation, the Standard Chartered Bank (Hong Kong) but
a confirmed date for this bank is 1879, followed by The Hongkong and Shanghai Banking Corporation in 1877, the Mercantile Bank in 1948 and the Bank of China in
1994.

 The Specimens are known from the Agra and Masterman's Bank and the Asiatic Banking Corporation between 1862-66.

 The National Bank of China issued theirs in the 1890s.

 There was a continuous issue till the Second World War in different colours and dimensions, they were reissued from 1946.

 The Mercantile bank ceased issue of this denomination after 1959.

 There was a standardisation of size in 1979 when the Chartered Bank reduced the size to that similar to HSBC.

 The colour was made uniform in 2003 when brown for all banknotes was adopted.
</item 8>

<item 9>
Bank Street (Hong Kong)
Bank Street () is a short street in Central, Hong Kong.

 It links Des Voeux Road Central to Queen's Road Central.

 It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong)
respectively.
</item 9>

<item 10>
HSBC lions
Various headquarters and branch buildings of The Hongkong and Shanghai Banking Corporation and the HSBC Group, into which the Hongkong and Shanghai Banking
Corporation has evolved, feature a pair of lion sculptures.

 The HSBC lions have become distinctive landmarks in their own right in Hong Kong and Shanghai respectively, with a further pair to be found in London.
</item 10>
=== END ITEM id=b218fe88c21f2a1a ===

=== ITEM id=3252409a332680ba ===
Gold sentences and facts:
Sentence 0: Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.
  Fact 0.0: Eriogonum is the scientific name for a genus of flowering plants.
  Fact 0.1: The genus Eriogonum is in the family Polygonaceae.
Sentence 1: Aloinopsis is a genus of ice plants from South Africa.
  Fact 1.0: Aloinopsis is a genus of ice plants.
  Fact 1.1: The Aloinopsis ice plants come from South Africa.

Retrieved items:
<item 1>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 1>

<item 2>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 2>

<item 3>
Graph query row:
entity: Aloinopsis
relationship: PART_OF
related: ice plants
</item 3>

<item 4>
Alismataceae
The water-plantains (Alismataceae) are a family of flowering plants, comprising 11 genera and between 85 and 95 species.

 The family has a cosmopolitan distribution, with the greatest number of species in temperate regions of the Northern Hemisphere.

 Most of the species are herbaceous aquatic plants growing in marshes and ponds.
</item 4>

<item 5>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 5>

<item 6>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 6>

<item 7>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 7>

<item 8>
Aloinopsis
Aloinopsis is a genus of ice plants from South Africa.
</item 8>

<item 9>
Graph query row:
entity: Eriogonum
relationship: PART_OF
related: Polygonaceae
</item 9>

<item 10>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 10>
=== END ITEM id=3252409a332680ba ===

=== ITEM id=244aed629c9e0110 ===
Gold sentences and facts:
Sentence 0: Dean Mills Reservoir is a small water reservoir to the north of the town of Bolton, Greater Manchester located high on the slopes of Winter Hill.
  Fact 0.0: Dean Mills Reservoir is a small water reservoir.
  Fact 0.1: Dean Mills Reservoir is north of the town of Bolton, Greater Manchester.
  Fact 0.2: Dean Mills Reservoir is located high on the slopes of Winter Hill.
Sentence 1:  It is located on Rivington Moor, Chorley and is 1496 ft high.
  Fact 1.0: Dean Mills Reservoir is located on Rivington Moor, Chorley.
  Fact 1.1: Dean Mills Reservoir is 1496 ft high.

Retrieved items:
<item 1>
Fisher Hill Reservoir
The Fisher Hill Reservoir and Gatehouse are a historic element of the public water supply for the Greater Boston area.

 The reservoir was located on Fisher Road between Hyslop and Channing Roads in Brookline, Massachusetts, and is now the site of Fisher Hill Reservoir Park.

 It was built in 1886-87 as an early component of the Boston Water Board's expansion of its high service system.

 The gatehouse may have been designed by Arthur Vinal, who also designed the high pumping station at Chestnut Hill Reservoir.

 It is a two-story Richardsonian Romanesque structure, with its first floor finished in stone and its second in brick.

 Brownstone trim is used on the windows and corner quoins, and the voussoirs which form the arches on the first floor.

 There are pipes to the reservoir and down to Chestnut Hill, and gates for controlling access to local the distribution network.

 The building was taken out of service in the 1950s.
</item 1>

<item 2>
Arnold Mills Historic District
Arnold Mills Historic District is a historic district encompassing a modest 19th-century mill village in eastern Cumberland, Rhode Island.

 The district lies along the Nate Whipple Highway (Rhode Island Route 120) and Sneech Pond Road, south of the Arnold Mills Reservoir.

 Sneech Pond Road was formerly the major east-west highway through the area prior to the construction of the Nate Whipple Highway in the 1960s.

 The Arnold Mills village is in part bisected by Abbott Run, the stream which serves as the outlet of the reservoir; Sneech Pond Road crosses the run on an
early 20th-century steel Pratt pony truss, now closed to vehicular traffic.

 The houses along this road generally date from the late 18th to mid-19th century, and mainly reflect Federal and Greek Revival styling.

 The most prominent structure in the district is the Arnold Mills United Methodist Church, located at the western end of the district on Nate Whipple Highway;
it was built 1825-27 and remodeled in 1846.
</item 2>

<item 3>
Toorourrong Reservoir
Toorourrong Reservoir is a small water supply reservoir located on the southern slopes of the Great Dividing Range approximately 40 km north of Melbourne,
Victoria, Australia.

 The reservoir is formed by the Toorourrong Dam across the Plenty River, and an interbasin transfer.

 The dam is operated by Melbourne Water and the reservoir forms part of the Melbourne water supply system.

 Water from the Toorourrong Reservoir flows by aqueduct to the Yan Yean Reservoir.
</item 3>

<item 4>
Crystal Lake (Gilmanton, New Hampshire)
Crystal Lake is a 455 acre water body located in Belknap County in the Lakes Region of central New Hampshire, United States, in the town of Gilmanton.

 Crystal Lake is at the head of the Suncook River watershed.

 Water flows into Crystal Lake from the Belknap Range through Manning and Sunset lakes to the north.

 Water from Manning Lake enters Crystal Lake via Nelson Brook.

 Other waterways flowing into Crystal Lake include Wasson Brook and Mill Brook.

 Water flows out of a dam from the southern end of Crystal Lake via the Suncook Lakes and Suncook River to the Merrimack River.

 Shoreline development along Crystal Lake consists primarily of summer cottages, with a few year-round residents.

 Belknap Mountain and Mount Major can be seen from the shoreline.
</item 4>

<item 5>
Oakland Mills, New Jersey
Oakland Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The area is named for Oakland Mill, a mill along the Manalapan Brook operated by the Boo family.

 Some areas within the Manalapan Brook valley and surround forestland are preserved while the remainder of the area surrounding Oakland Mills has since been
developed into housing developments.
</item 5>

<item 6>
Winter Hill (North West England)
Winter Hill is a hill on the border of the boroughs of Chorley, Blackburn with Darwen and Bolton, in North West England.

 It is located on Rivington Moor, Chorley and is 1496 ft high.

 Part of the West Pennine Moors, it is a popular walking area, and has been the site of mining activity, aeroplane disasters and murders.
</item 6>

<item 7>
Dean Mills Reservoir
Dean Mills Reservoir is a small water reservoir to the north of the town of Bolton, Greater Manchester located high on the slopes of Winter Hill.
</item 7>

<item 8>
Minster Pool
Minster Pool is a reservoir located between Bird Street and Dam Street in the heart of the city of Lichfield, Staffordshire in the United Kingdom.

 The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.

 The pool was originally formed in the 11th century when a boggy stream was dammed at its eastern end to drive a mill on Dam Street.

 The pool was used as a mill pond and fishery until 1856 when the mill was demolished; it has since been retained for public amenity.
</item 8>

<item 9>
Taylors Mills, New Jersey
Taylors Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The main road that runs through the area is named for the community, Taylors Mills Road.
</item 9>

<item 10>
Moel y Gydros
Moel y Gydros (English: Bare Hill of Gydros ) is a hill located just outside the Snowdonia National Park on the Gwynedd / Conwy border in North Wales.

 The B4501, Frongoch to Cerrigydrudion road skirts the hills lower slopes.
</item 10>
=== END ITEM id=244aed629c9e0110 ===

=== ITEM id=7955ce1bb6253955 ===
Gold sentences and facts:
Sentence 0: Baghdad ER is a documentary released by HBO on May 21, 2006.
  Fact 0.0: Baghdad ER is a documentary.
  Fact 0.1: Baghdad ER was released by HBO.
  Fact 0.2: Baghdad ER was released on May 21, 2006.
Sentence 1: Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium in Berlin, Germany.
  Fact 1.0: Olympia is a documentary film.
  Fact 1.1: Olympia was released in 1938.
  Fact 1.2: Olympia is a German film.
  Fact 1.3: Olympia was written by Leni Riefenstahl.
  Fact 1.4: Olympia was directed by Leni Riefenstahl.
  Fact 1.5: Olympia was produced by Leni Riefenstahl.
  Fact 1.6: Olympia documents the 1936 Summer Olympics.
  Fact 1.7: The 1936 Summer Olympics were held in the Olympic Stadium in Berlin, Germany.

Retrieved items:
<item 1>
Matthew O'Neill (filmmaker)
Matthew O'Neill is a documentary filmmaker best known for his work on the HBO film "Baghdad ER", for which he and co-creator Jon Alpert won three Emmy Awards.
</item 1>

<item 2>
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
</item 2>

<item 3>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 3>

<item 4>
Shamal Sabri
He has proven himself over and over as an important player in Kurdistan's developing film industry and eventually earned the position of Artistic Director of
Duhok IFF in 2016.

 Shamal is the head of submission committee for films representing Iraq in the Asia Pacific Screen Awards.
</item 4>

<item 5>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 5>

<item 6>
Anatomy (film)
Anatomy (German: "Anatomie" ) is a 2000 German horror film written and directed by Stefan Ruzowitzky that stars Franka Potente.

 A sequel, "Anatomy 2" ("Anatomie 2") was released in 2003.

 The film was a box-office success in Germany, and Columbia Pictures released the film's English-dubbed version in the United States theatrically.
</item 6>

<item 7>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 7>

<item 8>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 8>

<item 9>
Baghdad Jewish Arabic
Baghdad Jewish Arabic is the Arabic dialect spoken by the Jews of Baghdad and other towns of Southern Iraq.

 This dialect differs from the dialect spoken by the Jews in Northern Iraq, such as Mosul and 'Ana.

 The Baghdadi and Northern dialects may be regarded as subvarieties of Judeo-Iraqi Arabic.

 As with most Judeo-Arab communities, there are likely to be few, if any, speakers of the Judeo-Iraqi Arabic dialects who still reside within Iraq.

 Rather these dialects have been maintained or are facing critical endangerment within respective Judeo-Iraqi diasporas, namely those of Israel and the United
States.

 In 2014, the film "Farewell Baghdad" (Arabic: مطير الحمام; Hebrew: מפריח היונים, lit.

 "The Dove Flyer"), which is performed mostly in Jewish Baghdadi Arabic dialect, became the first film to be almost completely performed in Judeo-Iraqi Arabic.
</item 9>

<item 10>
Baghdad ER
Baghdad ER is a documentary released by HBO on May 21, 2006.

 It shows the Iraq war from the perspective of a military hospital in Baghdad.

 It has some relatively disturbing scenes in it (e.g. amputations), therefore the U.S. Army is officially warning that military personnel watching it could
experience symptoms of post-traumatic stress disorder (PTSD).
</item 10>
=== END ITEM id=7955ce1bb6253955 ===

=== ITEM id=1a3fe984e5849375 ===
Gold sentences and facts:
Sentence 0: Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.
  Fact 0.0: Isaac Asimov was born Isaak Ozimov.
  Fact 0.1: Isaac Asimov was born on January 2, 1920.
  Fact 0.2: Isaac Asimov's dates run from January 2, 1920 to April 6, 1992.
  Fact 0.3: Isaac Asimov was a Russian-American writer.
  Fact 0.4: Isaac Asimov was a professor of biochemistry at Boston University.
Sentence 1: Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.
  Fact 1.0: Michael Cunningham was born on November 6, 1952.
  Fact 1.1: Michael Cunningham is a U.S. novelist.
  Fact 1.2: Michael Cunningham is a U.S. screenwriter.

Retrieved items:
<item 1>
The Last Question
"The Last Question" is a science fiction short story by American writer Isaac Asimov.

 It first appeared in the November 1956 issue of "Science Fiction Quarterly" and was anthologized in the collections "Nine Tomorrows" (1959), "The Best of Isaac
Asimov" (1973), "Robot Dreams" (1986), the retrospective "Opus 100" (1969), and in "Isaac Asimov: The Complete Stories, Vol.

 1" (1990).

 It was Asimov's favorite short story of his own authorship, and is one of a loosely connected series of stories concerning a fictional computer called
Multivac.

 The story overlaps science fiction, theology, and philosophy.
</item 1>

<item 2>
Isaac Asimov
Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.

 He was known for his works of science fiction and popular science.

 Asimov was a prolific writer, and wrote or edited more than 500 books and an estimated 90,000 letters and postcards. His books have been published in 9 of the
10 major categories of the Dewey Decimal Classification.
</item 2>

<item 3>
Isaac Asimov's Magical Worlds of Fantasy
Isaac Asimov's Magical Worlds of Fantasy is a series of twelve themed paperback fantasy and science fiction anthologies edited by Isaac Asimov, Martin H.
Greenberg and Charles G. Waugh, a companion set to the ten volume "Isaac Asimov's Wonderful Worlds of Science Fiction", produced by the same editors.

 It was published by Signet/New American Library from 1983 to 1991.

 Volumes 1 and 2 were also issued in hardcover in an omnibus collection titled "Isaac Asimov's Magical Worlds of Fantasy: Witches & Wizards".
</item 3>

<item 4>
Isaac Asimov's Robot City
Isaac Asimov's Robot City is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 It takes place between "The Robots of Dawn" and "Robots and Empire".

 Each volume is complete in itself, but they form a continuing series.

 The novels were written in response to a writing challenge issued by Asimov to write a series involving the Three Laws of Robotics, which brought about a
collaboration of several authors.

 Asimov provided outlines for stories which filled in the gap between Asimov's own robot stories and his "Foundation" series, explaining the disappearance of
the robots prior to the establishment of the galactic empire. "

Isaac Asimov's Robots and Aliens" followed in this series, with the same protagonists and many other characters.

 The common theme of all books of both series is the interaction between the characters and autonomous cities run and populated by robots (the "robot cities" of
the series title).

 Robot City was also released as a mystery game for the PC in 1995.

 The player takes the role of Derec.
</item 4>

<item 5>
Isaac Asimov's Robot City: Odyssey
Isaac Asimov's Robot City: Odyssey is a young adult science fiction novel written in 1987 by Michael P. Kube-McDowell.

 It is part of the series Isaac Asimov's Robot City, inspired by Isaac Asimov's "Robot" series.

 The 1995 computer game Robot City is based on the plot of "Odyssey".
</item 5>

<item 6>
Michael Cunningham
Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.

 He is best known for his 1998 novel "The Hours", which won the Pulitzer Prize for Fiction and the PEN/Faulkner Award in 1999.

 Cunningham is a senior lecturer of creative writing at Yale University.
</item 6>

<item 7>
Wandering Stars
Wandering Stars is an anthology of Jewish fantasy and science fiction, edited by American writer Jack Dann, originally published by Harper & Row in 1974.

 It represented, according to the book cover, "the first time in science fiction that the Jew - and the richness of his themes and particular points of view --
will appear without a mask."

 In his introduction, "Why Me?"

, Isaac Asimov discussed how many Jewish science fiction writers prior to that time had used gentile pen names in order to get published: "Many of the Jewish
pulp writers, however, used pen names as a matter of sound business.

 A story entitled "War Gods of the Oyster-Men of Deneb" did not carry conviction if it was written by someone named Chaim Itzkowitz."

 He then goes on to discuss the pen names of various Jewish writers included in this book.

 "Wandering Stars" is therefore of historical significance as the first science fiction anthology where Jewish writers openly identified themselves as such.

 It was followed by a second anthology, "More Wandering Stars", also edited by Jack Dann, published by Doubleday in 1981.
</item 7>

<item 8>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 8>

<item 9>
Isaac Asimov's Robot City: Perihelion
Isaac Asimov's Robot City: Perihelion is a book written in 1988 by William F. Wu.

 It is part of the series Isaac Asimov's Robot City, which was inspired by Isaac Asimov's "Robot" series.
</item 9>

<item 10>
Isaac Asimov's Robots and Aliens
Isaac Asimov's Robots and Aliens is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 Each volume is complete in itself, but they form a continuing series.

 The series follows the action of the novels of the "Isaac Asimov's Robot City" series, with the same protagonists Derec and Ariel, and many other characters.

 The plot deals with the Three Laws and encounters between robots and different varieties of alien life.
</item 10>
=== END ITEM id=1a3fe984e5849375 ===

=== ITEM id=1e9a981aff957897 ===
Gold sentences and facts:
Sentence 0: Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.
  Fact 0.0: Heavenly Creatures is a 1994 New Zealand psychological drama.
  Fact 0.1: Heavenly Creatures was directed by Peter Jackson.
  Fact 0.2: Peter Jackson co-wrote the screenplay with his partner, Fran Walsh.
  Fact 0.3: Heavenly Creatures is about the notorious 1954 Parker-Hulme murder case.
  Fact 0.4: The Parker-Hulme murder case took place in Christchurch, New Zealand.
Sentence 1:  The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.
  Fact 1.0: The main premise of Heavenly Creatures deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme.
  Fact 1.1: Pauline Parker and Juliet Hulme murder Pauline Parker's mother.
Sentence 2: Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk series.
  Fact 2.0: Anne Perry was born on 28 October 1938 as Juliet Marion Hulme.
  Fact 2.1: Anne Perry is an English author of historical detective fiction.
  Fact 2.2: Anne Perry is best known for her Thomas Pitt and William Monk series.

Retrieved items:
<item 1>
Heavenly Creatures
Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the
notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.

 The film features Melanie Lynskey and Kate Winslet in their screen debuts with supporting roles by Sarah Peirse, Diana Kent, Clive Merrison, and Simon
O'Connor.

 The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.

 The events of the film cover the period from their meeting in 1952 to the murder in 1954.
</item 1>

<item 2>
Kate Winslet, a British actress and singer, debuted as Juliet Hulme in *Heavenly Creatures* (1994). Her early films included *Sense and Sensibility* (1995),
earning a Best Supporting Actress Oscar nomination; *Jude* and *Hamlet* (1996); and *Titanic* (1997), which brought Best Actress nominations from the Academy,
Golden Globe, and Screen Actors Guild Awards.
</item 2>

<item 3>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 3>

<item 4>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 4>

<item 5>
The entries profile performers and productions, including *Doctor Who*, *EastEnders* and *Waterloo Road*. Juliet Weber is a producer, director, writer and
editor. Juno Temple starred in *Vinyl*; Holly Hunter won major awards for *The Piano*. Kathryn Drysdale played Louise Brooks and Meghan Markle. Kate Winslet
rose to prominence with *Heavenly Creatures* and *Titanic*. Cate Blanchett won two Oscars, including for *The Aviator*. Reese
</item 5>

<item 6>
Juno Temple (born 21 July 1989) is an English actress known for films including *Black Mass*, *The Other Boleyn Girl*, *Wild Child*, *Atonement*, *Maleficent*,
*The Three Musketeers* and *The Dark Knight Rises*. She starred in HBO’s period drama *Vinyl*.
</item 6>

<item 7>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 7>

<item 8>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 8>

<item 9>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 9>

<item 10>
Parker–Hulme murder case
The Parker–Hulme murder case began in the city of Christchurch, New Zealand, on 22 June 1954, when Honora Rieper (also known as Honora Parker, her legal name)
was killed by her teenaged daughter, Pauline Parker, and Pauline's close friend Juliet Hulme (later known as Anne Perry).

 Parker was 16 at the time, while Hulme was 15.
</item 10>
=== END ITEM id=1e9a981aff957897 ===

=== ITEM id=aee1b993d254049e ===
Gold sentences and facts:
Sentence 0: The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.
  Fact 0.0: The 1892 Auburn Tigers football team represented Auburn University.
  Fact 0.1: The 1892 Auburn Tigers football team played in the 1892 college football season.
Sentence 1:  It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.
  Fact 1.0: The 1892 Auburn Tigers football team was the first college football team fielded by the Agricultural and Mechanical College of Alabama.
  Fact 1.1: The Agricultural and Mechanical College of Alabama is now known as Auburn University.
Sentence 2:  With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in Alabama.
  Fact 2.0: Auburn University has more than 22,000 undergraduate students.
  Fact 2.1: Auburn University has a total enrollment of more than 28,000.
  Fact 2.2: Auburn University has 1,260 faculty members.
  Fact 2.3: Auburn University is the second largest university in Alabama.

Retrieved items:
<item 1>
1899 Connecticut Aggies football team
The 1899 Connecticut Aggies football team represented Connecticut Agricultural College, now the University of Connecticut, in the 1899 college football season.

 This was the fourth year that the school fielded a football team.

 The Aggies were led by first year head coach T. D. Knowles, and completed the season with a record of 6–2.
</item 1>

<item 2>
West Alabama Tigers football
The West Alabama Tigers football program is the intercollegiate American football team for the University of West Alabama located in the U.S. state of Alabama.

 The team competes in the NCAA Division II and are members of the Gulf South Conference.

 West Alabama's first football team was fielded in 1938.

 The team plays its home games at the 7,000 seat Tiger Stadium in Livingston, Alabama.

 The Tigers are coached by Brett Gilliland.
</item 2>

<item 3>
1905 Alabama Crimson White football team
The 1905 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1905 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 13th overall and 10th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach Jack Leavenworth, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of six wins and four losses (6–4 overall, 4–4 in the SIAA).
</item 3>

<item 4>
1983 Alabama Crimson Tide football team
The 1983 Alabama Crimson Tide football team (variously "Alabama", "UA", "Bama" or "The Tide") represented the University of Alabama in the 1983 NCAA Division
I-A football season.

 It was the Crimson Tide's 91st overall and 50th season as a member of the Southeastern Conference (SEC).

 The team was led by head coach Ray Perkins, in his first year, and played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in
Birmingham, Alabama.

 They finished the season with a record of eight wins and four losses (8–4 overall, 4–2 in the SEC).

 Ray Perkins, who played as a wide receiver for Bear Bryant in the 1960s, was named as the new head coach at Alabama on December 14, 1982, to succeed Bryant
after his 25-year tenure as Alabama's head coach.
</item 4>

<item 5>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 5>

<item 6>
1997 Alabama Crimson Tide football team
The 1997 Alabama Crimson Tide football team represented the University of Alabama in the college football season of 1997–98.

 The team's head coach was Mike DuBose, who was entering his first year at Alabama.

 They played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in Birmingham, Alabama and competed in the West Division of the
Southeastern Conference.

 Alabama would finish with a record of 4–7 record in suffering the program's first losing season since the 1984 season.

 The loss against Kentucky marked Alabama's first ever overtime game, as overtime rules for college football had gone into effect the previous season.
</item 6>

<item 7>
1902 Connecticut Aggies football team
The 1902 Connecticut Aggies football team represented Connecticut Agricultural College, now the University of Connecticut, in the 1902 college football season.

 This was the seventh year that the school fielded a football team.

 The Aggies were led by first year head coach Edwin O. Smith, and completed the season with a record of 4–3.
</item 7>

<item 8>
1906 Alabama Crimson White football team
The 1906 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1906 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 14th overall and 11th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach J. W. H. Pollard, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of five wins and one loss (5–1 overall, 3–1 in the SIAA).
</item 8>

<item 9>
Auburn University
Auburn University (AU or Auburn) is a public research university in Auburn, Alabama, United States.

 With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in
Alabama.

 Auburn University is one of the state's two public flagship universities.
</item 9>

<item 10>
1892 Auburn Tigers football team
The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.

 It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.

 The squad was first coached by George Petrie.

 Auburn shut out Georgia Tech, 26–0, just two days after being shut out themselves by North Carolina, 64–0.

 The team finished the season with a record of 2–2.
</item 10>
=== END ITEM id=aee1b993d254049e ===
