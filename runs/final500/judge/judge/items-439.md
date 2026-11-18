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

=== ITEM id=d99adf8e3ab2b8f3 ===
Gold sentences and facts:
Sentence 0: Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and Randall Counties, Texas, United States.
  Fact 0.0: Rick Husband Amarillo International Airport has the IATA code AMA.
  Fact 0.1: Rick Husband Amarillo International Airport has the ICAO code KAMA.
  Fact 0.2: Rick Husband Amarillo International Airport has the FAA LID AMA.
  Fact 0.3: Rick Husband Amarillo International Airport is a public airport.
  Fact 0.4: Rick Husband Amarillo International Airport is six miles (10 km) east of downtown Amarillo.
  Fact 0.5: Rick Husband Amarillo International Airport is in Potter and Randall Counties, Texas, United States.
Sentence 1: Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County, North Dakota.
  Fact 1.0: Grand Forks International Airport has the IATA code GFK.
  Fact 1.1: Grand Forks International Airport has the ICAO code KGFK.
  Fact 1.2: Grand Forks International Airport has the FAA LID GFK.
  Fact 1.3: Grand Forks International Airport is a public airport.
  Fact 1.4: Grand Forks International Airport is five miles (8 km) northwest of Grand Forks.
  Fact 1.5: Grand Forks International Airport is in Grand Forks County, North Dakota.

Retrieved items:
<item 1>
Grand Forks International
The Grand Forks International (GFI) is an annual international invitational baseball tournament hosted at James Donaldson Park in Grand Forks, British Columbia.

 The GFI is the largest invitational baseball tournament in Canada and is a large part of the culture in Grand Forks.

 The tournament is also notable for being run entirely by volunteers.
</item 1>

<item 2>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 2>

<item 3>
H-T Ranch
H-T Ranch is a historic ranch complex 10 mi west of Amidon, North Dakota.

 The ranch originally consisted of ten buildings; however, only the ranch house and log barn survive.

 The ranch served as the headquarters of the Little Missouri Cattle Company, which was run by Arthur Clark Hidekoper.

 Hidekober established the ranch in the 1880s, and by the end of the decade, it had become the "most notable" ranch in the area.

 The ranch house, called Shackford, was built in 1896; its uncommon style resembles a bungalow but also borrows from other styles such as the Stick style.

 By 1906, the "Fargo Forum" described the ranch as "the biggest and most important [ranch] in the state" and "one of the largest horse raising outlets in the
world".

 In the same year, Hidekoper sold the 70,000 acre ranch; the sale was the largest land deal in North Dakota history.

 After the sale, a land company reduced the ranch to 5000 acre ; it was later used as a dude ranch in the 1920s.
</item 3>

<item 4>
Downtown Grand Forks
Downtown Grand Forks is the original commercial center of Grand Forks, North Dakota.

 Located on the western bank of the Red River of the North, the downtown neighborhood is situated near the fork of the Red River and the Red Lake River.

 While downtown is no longer the dominant commercial area of the Greater Grand Forks community, it remains the historic center of Grand Forks.

 An 80.4 acre portion was listed on the National Register of Historic Places in 2005, as Downtown Grand Forks Historic District.

 Today, downtown Grand Forks is home to many offices, stores, restaurants, and bars.
</item 4>

<item 5>
Grand Forks International Airport
Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County,
North Dakota.

 GFK has no scheduled passenger flights out of the country but has an "international" title (like many other airports) because it has customs service for
arrivals from Canada and other countries.
</item 5>

<item 6>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 6>

<item 7>
Emerado, North Dakota
Emerado is a city in Grand Forks County, North Dakota, United States located near Grand Forks Air Force Base.

 It is part of the "Grand Forks, ND-MN Metropolitan Statistical Area" or "Greater Grand Forks".

 The population was 414 at the 2010 census.

 Grand Forks Air Force Base is located near Emerado.
</item 7>

<item 8>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 8>

<item 9>
John D. Odegard School of Aerospace Sciences
The John D. Odegard School of Aerospace Sciences (UND Aerospace) is a multidisciplinary college within the University of North Dakota (UND) in Grand Forks,
North Dakota.

 The school was formed in 1968.

 The majority of the school's fleet of over 120 aircraft is based at nearby Grand Forks International Airport and is the largest fleet of civilian flight
training aircraft in North America.

 UND Aerospace also operates flight training centers in Crookston, Minnesota, and Phoenix, Arizona.

 Today, the school has many aerospace-related programs including commercial aviation (fixed wing and rotorcraft), air traffic control, airport management, Space
Studies, Computer Science, Atmospheric Sciences, and Earth System Science & Policy.

 Currently, the school has over 500 faculty and 1,900 students making it the second largest of UND’s degree-granting colleges.

 The present dean of the school is Dr. Paul Lindseth.
</item 9>

<item 10>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 10>
=== END ITEM id=d99adf8e3ab2b8f3 ===

=== ITEM id=0c94eb82dc39c186 ===
Gold sentences and facts:
Sentence 0:  They have produced two studio albums and an EP.
  Fact 0.0: They have produced two studio albums.
  Fact 0.1: They have produced an EP.
Sentence 1: Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.
  Fact 1.0: Motion City Soundtrack was an American rock band.
  Fact 1.1: Motion City Soundtrack formed in Minneapolis, Minnesota.
  Fact 1.2: Motion City Soundtrack formed in 1997.
Sentence 2:  Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.
  Fact 2.0: Motion City Soundtrack had a nearly twenty-year career.
  Fact 2.1: Motion City Soundtrack toured heavily over the course of its career.
  Fact 2.2: Motion City Soundtrack released six studio albums.
  Fact 2.3: The majority of Motion City Soundtrack's six studio albums were released on the independent label Epitaph Records.

Retrieved items:
<item 1>
Commit This to Memory
Commit This to Memory is the second studio album by American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on June 7, 2005, in the United States by Epitaph Records.

 Motion City Soundtrack, formed in 1997, had first found success with their debut album, "I Am the Movie" (2002).

 The band toured in the interim years, creating positive word-of-mouth.

 In 2004, the band joined Blink-182 on the road for a string of shows, which led to their bassist, Mark Hoppus, joining the band in the studio for his first
producing effort.
</item 1>

<item 2>
Candlelight Red
Candlelight Red is a rock band from Williamsport, Pennsylvania.

 They have produced two studio albums and an EP.

 Their EP "Demons" and album "Reclamation" was produced by Morgan Rose of Sevendust.
</item 2>

<item 3>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 3>

<item 4>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 4>

<item 5>
Even If It Kills Me
Even if It Kills Me is the third studio album by American rock band Motion City Soundtrack.

 Produced by Ric Ocasek, Adam Schlesinger and Eli Janney, the album was released on September 18, 2007, in the United States by Epitaph Records.

 Motion City Soundtrack, based in Minneapolis, Minnesota, made a breakthrough with their second album, "Commit This to Memory", garnering praise and independent
buzz upon its 2005 release.

 Following the release, the band toured relentlessly over the next two years, during which time frontman Justin Pierre struggled with alcohol and substance
abuse.

 These addictions were infused into the writing process of "Even If Kills Me", which was recorded in early 2007 Stratosphere Studios in Chelsea and the
legendary Electric Lady Studios in Greenwich Village.
</item 5>

<item 6>
Motion City Soundtrack discography
The following is the discography of Motion City Soundtrack, an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 6>

<item 7>
List of songs recorded by Motion City Soundtrack
American rock band Motion City Soundtrack recorded songs for six studio albums, various soundtracks, compilations, and non-album singles.

 The majority of the group's original material was written by vocalist and guitarist Justin Pierre.

 The band also recorded cover versions of other artists' songs at various points throughout their career, including by the Beatles, Limbeck, the Police, R.E.M.,
the Rentals, Rilo Kiley, Trampled by Turtles, and the Weakerthans.

 In all, the group recorded 107 songs, nine of which were covers.
</item 7>

<item 8>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 8>

<item 9>
Back to the Beat (EP)
Back to the Beat was the first significant EP release by the Minneapolis based pop punk band Motion City Soundtrack.

 The EP was recorded in Sound In Motion Recordings and released by Modern Radio Records and includes tracks that were later re-released.

 "Throw Down", "Back to the Beat" and "Capital H" were re-released on a split EP with Schatzi.

 "Capital H" was re-released on Motion City Soundtrack's first album "I Am the Movie".

 At the time of recording, the band members differed from the Motion City Soundtrack line-up formed in 2002.

 They were Joshua Cain on guitar and vocals, Austin Lindstrom on bass guitar, Joel Habedank on drums and Justin Pierre on lead vocals, guitar and keyboard.
</item 9>

<item 10>
My Dinosaur Life
My Dinosaur Life is the fourth studio album by the American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on January 19, 2010 by Columbia Records.

 After many years on independent label Epitaph Records, Motion City Soundtrack signed to major label Columbia in 2006, prior to the release of their third
album, "Even If It Kills Me" (2007).

 Blink-182 bassist Mark Hoppus returned to produce the album; he had previously worked with the band on their 2005 breakthrough "Commit This to Memory".
</item 10>
=== END ITEM id=0c94eb82dc39c186 ===

=== ITEM id=7b79f92cadf39065 ===
Gold sentences and facts:
Sentence 0:  Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the summer.
  Fact 0.0: Obertraun is a popular holiday destination.
  Fact 0.1: Obertraun offers skiing and snowboarding in the winter.
  Fact 0.2: Obertraun offers mountain biking in the summer.
  Fact 0.3: Obertraun offers swimming and kayaking in the summer.
Sentence 1:  The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the precious salt mines of the Habsburg Monarchy.
  Fact 1.0: The name 'Salzkammergut' literally means 'Estate of the Salt Chamber'.
  Fact 1.1: The name 'Salzkammergut' derives from the Imperial Salt Chamber.
  Fact 1.2: The Imperial Salt Chamber was the authority charged with running the precious salt mines of the Habsburg Monarchy.

Retrieved items:
<item 1>
Aspen/Snowmass
Aspen Snowmass is a winter resort complex located in Pitkin County in western Colorado in the United States.

 Owned and operated by the Aspen Skiing Company it comprises four skiing/snowboarding areas on four adjacent mountains in the vicinity of the towns of Aspen and
Snowmass Village.

 The four areas collectively form one of the most famous winter resorts in the world and are annually the destination for visitors from all over the world.
</item 1>

<item 2>
Salzkammergut
The Salzkammergut is a resort area located in Austria.

 It stretches from the City of Salzburg eastwards along the Austrian Alpine Foreland and the Northern Limestone Alps to the peaks of the Dachstein Mountains,
spanning the federal states of Upper Austria, Salzburg, and Styria.

 The main river of the region is the Traun, a right tributary of the Danube.

 The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the
precious salt mines of the Habsburg Monarchy.

 It is a UNESCO World Heritage Site.
</item 2>

<item 3>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 3>

<item 4>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 4>

<item 5>
Obertraun
Obertraun, Upper Austria is a village in the Salzkammergut, a region in Austria.

 It is located near the Hallstätter See (Hallstatt Lake) and Hoher Dachstein.

 Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the
summer.
</item 5>

<item 6>
Hundsheimer Berg
Hundsheimer Berg is a hill located in the Hundsheimer Berge hill range in Lower Austria, Austria close to the border with Slovakia.

 Its peak is 481 m above sea level which makes it the highest hill in the Hundsheimer Berge.

 The north slopes of the Hundsheimer Berg are covered with forests while the southern slopes have a steppe vegetation.

 Since 1965, a nature reserve with the same name, "Hundsheimer Berg", is located on these southern slopes.
</item 6>

<item 7>
Appenzeller cheese
Appenzeller cheese is a hard cow's-milk cheese produced in the Appenzell region of northeast Switzerland.

 A herbal brine, sometimes incorporating wine or cider, is applied to the wheels of cheese while they cure, which flavors and preserves the cheese while
promoting the formation of a rind.
</item 7>

<item 8>
Appenzell Alps
The Appenzell Alps (German: "Appenzeller Alpen" ) are a mountain range in Switzerland on the northern edge of the Alps.

 They extend into the cantons of Appenzell Ausserrhoden, Appenzell Innerrhoden and St. Gallen and are bordered by the Glarus Alps to the west and the Rätikon to
the south-east.
</item 8>

<item 9>
List of Austrian mountain climbers
This is a list of Austrian mountaineers.

 On 5 of the 14 Eight-thousanders Austrians have made the first ascent, more than any other nation can claim.

 Out of the total of 9 Austrian mountaineers who made first ascents of Eight-thousanders, 3 have been members of the Edelweiss Club Salzburg, an association of
mountaineers founded in Salzburg in 1881
</item 9>

<item 10>
Graph query row:
entity: Salzkammergut
connection_path: ['Salzkammergut', 'Obertraun']
connection_relationships: ['LOCATED_IN']
related_relationship: LOCATED_IN
related_entity: Obertraun
</item 10>
=== END ITEM id=7b79f92cadf39065 ===

=== ITEM id=f365cf020fbcdcf6 ===
Gold sentences and facts:
Sentence 0: Saddle Rock Elementary School or simply "Saddle Rock" is an elementary school, comprising grades Kindergarten through 5.
  Fact 0.0: Saddle Rock Elementary School is also called Saddle Rock.
  Fact 0.1: Saddle Rock Elementary School is an elementary school.
  Fact 0.2: Saddle Rock Elementary School comprises grades Kindergarten through 5.
Sentence 1:  It is a public school located in Great Neck, New York, USA as part of the Great Neck School District.
  Fact 1.0: Saddle Rock Elementary School is a public school.
  Fact 1.1: Saddle Rock Elementary School is located in Great Neck, New York, USA.
  Fact 1.2: Saddle Rock Elementary School is part of the Great Neck School District.
Sentence 2: The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.
  Fact 2.0: The Great Neck School District is a comprehensive community public school district.
  Fact 2.1: The Great Neck School District primarily serves students in Great Neck, New York.
Sentence 3:  It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
  Fact 3.0: The Great Neck School District is Union Free School District Number 7.
  Fact 3.1: The Great Neck School District is in the Town of North Hempstead.
  Fact 3.2: The Great Neck School District is in Nassau County, New York, United States.
  Fact 3.3: The Great Neck School District is on Long Island.

Retrieved items:
<item 1>
Ridgefield Park station
Ridgefield Park Station, also known as West Shore Station, was railroad station in Ridgefield Park, New Jersey at the foot of Mount Vernon Street served by the
New York, Susquehanna and Western Railroad (NYSW) and the West Shore Railroad, a division of New York Central (NYCRR) The New York, Ontario and Western Railway
(NYO&W) had running rights along the West Shore and sometimes stopped at Ridgefield Park.

 First opened in 1883 it was one of three passenger stations in the village, the others being the Little Ferry Station to the south and Westview Station to the
north.

 The station house, built at a cost $100,000 opened in 1927.

 Southbound service crossed Overpeck Creek and continued to terminals on the Hudson River waterfront where there was connecting ferry service across the Hudson
River to Manhattan.

 Northbound near Bogota the parallel NYSW and West Shore lines diverge and continue into northern New Jersey, Pennsylviania, and upstate New York.

 Passenger service ended in 1966.
</item 1>

<item 2>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 2>

<item 3>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 3>

<item 4>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
</item 4>

<item 5>
Saddle Rock Elementary School
Saddle Rock Elementary School or simply "Saddle Rock" is an elementary school, comprising grades Kindergarten through 5.

 It is a public school located in Great Neck, New York, USA as part of the Great Neck School District.

 The school has been recognized by the Blue Ribbon Schools Program.
</item 5>

<item 6>
East Lyme High School
East Lyme High School is a high school located in the Flanders Village region of East Lyme, Connecticut.

 It is operated by East Lyme Public Schools.

 The mascot is Sven the Viking.

 Students from the town of Salem, Connecticut in grades 9 through 12 attend high school in East Lyme (as they have no high school of their own); this will be
the case until at least 2016 when the current co-op agreement between the two towns expires.
</item 6>

<item 7>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 7>

<item 8>
Village School (Great Neck, New York)
Village School (also known as VS) is an American long-established public alternative school.

 It is a member of Coalition of Essential Schools and is located in the village of Great Neck, New York, serving students in grades 8 through 12.

 Village School is one of three high schools in the Great Neck School District, which includes Great Neck North High School and Great Neck South High School.

 Village School offers its 39 students an outdoor education program, college preparatory program, and inclusion of students with disabilities.
</item 8>

<item 9>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 9>

<item 10>
Great Neck, NY, encompasses nine North Hempstead villages/hamlets (~40,000 residents), five ZIP codes and one school district; Saddle Rock is its Blue Ribbon
K–5 school. Floral Park village had 15,863 residents in 2010. Old Country Road serves central Nassau shopping and partly borders Hempstead/North Hempstead. Long
Island’s 75-mile outer barrier separates lagoons from the Atlantic. The Nassau Herald serves Five Towns. Other entries describe
</item 10>
=== END ITEM id=f365cf020fbcdcf6 ===

=== ITEM id=61867b625ed9fe0d ===
Gold sentences and facts:
Sentence 0: Montague (Monty) Banks (15 July 1897 [registered on 18 July 1897] – 7 January 1950 born Mario Bianchi) was an Italian comedian and film director who achieved success in the United States and in England.
  Fact 0.0: Montague (Monty) Banks was born Mario Bianchi.
  Fact 0.1: Montague (Monty) Banks was born 15 July 1897.
  Fact 0.2: Montague (Monty) Banks' birth was registered on 18 July 1897.
  Fact 0.3: Montague (Monty) Banks died on 7 January 1950.
  Fact 0.4: Montague (Monty) Banks was an Italian comedian.
  Fact 0.5: Montague (Monty) Banks was an Italian film director.
  Fact 0.6: Montague (Monty) Banks achieved success in the United States.
  Fact 0.7: Montague (Monty) Banks achieved success in England.
Sentence 1: George Seaton (April 17, 1911 – July 28, 1979) was an American screenwriter, playwright, film director and producer, and theatre director.
  Fact 1.0: George Seaton was born April 17, 1911.
  Fact 1.1: George Seaton died July 28, 1979.
  Fact 1.2: George Seaton was an American screenwriter.
  Fact 1.3: George Seaton was an American playwright.
  Fact 1.4: George Seaton was an American film director.
  Fact 1.5: George Seaton was an American film producer.
  Fact 1.6: George Seaton was an American theatre director.

Retrieved items:
<item 1>
Chicken Every Sunday
Chicken Every Sunday is a 1949 American comedy film directed by George Seaton.

 The screenplay by Seaton and Valentine Davies is based on the 1944 play of the same title by Julius J. Epstein and Philip G. Epstein, which was based on the
memoir by Rosemary Taylor.
</item 1>

<item 2>
Mark Benton
Mark Benton (born 16 November 1965) is an English actor, known for his roles as Eddie in "Early Doors", Howard in "Northern Lights" and Martin Pond in
"Barbara".

 Benton has also starred in BBC One drama, "Waterloo Road" as maths teacher, Daniel Chalk from 2011 to 2013 and again in 2014.
</item 2>

<item 3>
George Seaton
George Seaton (April 17, 1911 – July 28, 1979) was an American screenwriter, playwright, film director and producer, and theatre director.
</item 3>

<item 4>
The Country Girl (1954 film)
The Country Girl is a 1954 American drama film directed by George Seaton and starring Bing Crosby, Grace Kelly, and William Holden.

 Adapted by George Seaton from Clifford Odets' 1950 play of the same name, the film is about an alcoholic has-been actor struggling with the one last chance
he's been given to resurrect his career.

 Seaton won the Academy Award for Best Writing, Adapted Screenplay.

 It was entered in the 1955 Cannes Film Festival.
</item 4>

<item 5>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 5>

<item 6>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 6>

<item 7>
Catsmeat Potter-Pirbright
Claude Cattermole "Catsmeat" Potter-Pirbright is a recurring fictional character from the Jeeves stories of British comic writer P. G. Wodehouse, being a member
of the Drones Club and a longtime school friend of Jeeves's master Bertie Wooster.

 An actor also known as Claude Cattermole on stage or Claude Pirbright from his birth name Claude Cattermole Pirbright, he has the nickname Catsmeat (i.e.
lights - lung-based cat food).
</item 7>

<item 8>
Monty Banks
Montague (Monty) Banks (15 July 1897 [registered on 18 July 1897] – 7 January 1950 born Mario Bianchi) was an Italian comedian and film director who achieved
success in the United States and in England.
</item 8>

<item 9>
Keep Your Seats, Please
Keep Your Seats, Please is a 1936 British comedy film directed by Monty Banks and starring George Formby, Florence Desmond and Alastair Sim.

 It marked the film debut of the child star Binkie Stuart.

 The film was made by Associated Talking Pictures.
</item 9>

<item 10>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 10>
=== END ITEM id=61867b625ed9fe0d ===

=== ITEM id=432003a63d9b22e4 ===
Gold sentences and facts:
Sentence 0: Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.
  Fact 0.0: The Dogo Cubano is also known as the Cuban Mastiff.
  Fact 0.1: The Dogo Cubano is also known as the Cuban Dogo.
  Fact 0.2: The Dogo Cubano is also known as the Cuban Dogge.
  Fact 0.3: The Dogo Cubano is an extinct dog breed.
  Fact 0.4: The Dogo Cubano is a dog breed from Cuba.
Sentence 1: The Dutch Shepherd is a herding dog of Dutch origin.
  Fact 1.0: The Dutch Shepherd is a herding dog.
  Fact 1.1: The Dutch Shepherd is a herding dog of Dutch origin.

Retrieved items:
<item 1>
Romanian Raven Shepherd Dog
The Romanian Raven Shepherd Dog is a very large Romanian livestock guardian dog, taxonomized within the second group of dog breeds - Pinscher and Schanuzer -
Molossoid Breeds - Swiss Mountain and Cattle Dogs, section 2.2: Molossoid Breeds - Mountain type.
</item 1>

<item 2>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 2>

<item 3>
Cuban Criollo
The Criollo Cubano originates from Spanish horses brought to Cuba by Diego Velázquez in 1751.

 There are four different breeds known as criollo: the Cubano de Paso, the Pinto Cubano, the Criollo de Trote and the Patibarcino.

 It is a small stout breed used by the Guajiro people.
</item 3>

<item 4>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 4>

<item 5>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 5>

<item 6>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 6>

<item 7>
Podenco Canario
Podenco Canario (In English: "Canary Islands Hound", "Canarian Warren Hound") is a breed of dog originally from the Canary Islands.

 The Podenco Canario is still used in the Canary Islands today in packs as a hunting dog, primarily used for rabbit hunting.

 The word "podenco" is Spanish for "hound".

 "Canario" means "Canarian" or "of the Canaries".
</item 7>

<item 8>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 8>

<item 9>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 9>

<item 10>
Dogo Cubano
Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.

 It was of the Bull Mastiff type and was used for dog fighting.
</item 10>
=== END ITEM id=432003a63d9b22e4 ===

=== ITEM id=6cf2c7e450f61a19 ===
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
Miracle Mile (Manhasset)
The Miracle Mile is a prominent shopping district in Manhasset on the North Shore of Long Island in Nassau County, New York.

 The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
</item 1>

<item 2>
Christian Dior SE
Christian Dior SE (] ), commonly known as Dior, is a French luxury goods company controlled and chaired by French businessman Bernard Arnault, who also heads
LVMH – the world's largest luxury group.

 Dior itself holds 42.36% shares of and 59.01% voting rights within LVMH.
</item 2>

<item 3>
Concord watch
Concord Watch Company is a Swiss luxury goods company that is part of the Movado group that owns Movado, Ebel, ESQ, Coach and Hugo Boss.

 Founded in 1908, Concord was purchased in 1970 by the North American Watch Company, which also distributed the Piaget and Corum lines of watches.

 From the late 1970s to the late 1980s, Concord produced what would become some of the most marketed and respected luxury quartz watches on the market.

 Concord watches became recognized in large cities as status symbols and were noted for their innovation and design.

 Flagship quartz models such as the Concord Centurion and Concord Delirium ranged from $2,000 to $20,000 surpassing the price of base automatic Rolex, Cartier
and Omega wristwatches.

 By the 1990s Concord watches fell out of style and grew obsolete amidst the re-branding of the company.

 While the brand still exists today, it never regained its market position nor visibility that it once had.
</item 3>

<item 4>
Tabcorp Park Menangle
Tabcorp Park Menangle is a harness racing track in Sydney.

 Their main races are the Miracle Mile Pace, the NSW Derby, the NSW Oaks, the Len Smith Mile and the qualifiers for the Miracle Mile, the Allied Express Sprint
and the Canadian Club Sprint.

 It replaced Harold Park as Sydney's premier harness track in 2010.

 Fred Hastings and Anthony Manton are two callers that have called at the track before.
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
Americana Manhasset
International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine,
Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van
Sweden.

 Peter Marino is Americana Manhasset's master architect.
</item 6>

<item 7>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 7>

<item 8>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 8>

<item 9>
Mouawad
Mouawad is a privately held Swiss and Emirati luxury goods company that makes fashion accessories, jewelry and retail company that sources diamonds and
gemstones, designs, manufacturers, and sells jewelry collections, objects of art, and luxury watches.

 The firm has headquarters in Geneva, Switzerland, with a Middle East headquarters at Jumeirah Lakes Towers in Dubai.

 Founded in 1891 in Beirut, Lebanon by David Mouawad, the firm is now led by 4th generation co-guardians Fred Mouawad, Alain Mouawad and Pascal Mouawad.

 The firm's jewelry and watch design and manufacturing divisions are located in Switzerland and Asia Pacific.
</item 9>

<item 10>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 10>
=== END ITEM id=6cf2c7e450f61a19 ===

=== ITEM id=ecae4153215468e7 ===
Gold sentences and facts:
Sentence 0: Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".
  Fact 0.0: Muppet Treasure Island is a 1996 American musical adventure comedy film
  Fact 0.1: Muppet Treasure Island is based on Robert Louis Stevenson's Treasure Island
Sentence 1: Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written by Thomas McCarthy.
  Fact 1.0: Million Dollar Arm is a 2014 American biographical sports drama film
  Fact 1.1: Million Dollar Arm was directed by Craig Gillespie
  Fact 1.2: Million Dollar Arm was produced by Walt Disney Pictures
  Fact 1.3: Million Dollar Arm was made from a screenplay written by Thomas McCarthy

Retrieved items:
<item 1>
Managed Money
Managed Money is a 1934 short comedy film directed by Charles Lamont.

 The film stars Frank Coghlan Jr. and Shirley Temple.

 It was also known as "Frolics of Youth" and "Measured Money".

 This was the second film in which Temple starred as Mary Lou.

 The film tells the story of Sonny and Sid prospecting for gold to pay for their military academy education.

 Sonny's sister, Mary Lou stows away in the back of their car.

 Eventually, they meet an inventor who helps them financially.
</item 1>

<item 2>
The Muppets (film)
The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.

 The film is directed by James Bobin, written by Jason Segel and Nicholas Stoller, produced by David Hoberman and Todd Lieberman, and stars Segel, Amy Adams,
Chris Cooper and Rashida Jones, as well as Muppet performers Steve Whitmire, Eric Jacobson, Dave Goelz, Bill Barretta, David Rudman, Matt Vogel, and Peter Linz.

 Bret McKenzie served as music supervisor, writing four of the film's five original songs, and Christophe Beck composed the film's score.

 In "The Muppets", devoted fan Walter, his brother Gary, and Gary's girlfriend Mary help Kermit the Frog reunite the disbanded Muppets, as they must raise $10
million to save the Muppet Theater from Tex Richman, a businessman who plans to demolish the studio to drill for oil.
</item 2>

<item 3>
Reaching for the Moon (1917 film)
Reaching for the Moon is a 1917 American silent adventure film directed by John Emerson and written by John Emerson, Joseph Henabery, and Anita Loos.

 The film stars Douglas Fairbanks, Eileen Percy, Richard Henry Cummings, Millard Webb, Eugene Ormonde, and Frank Campeau.

 The film was released on November 17, 1917, by Paramount Pictures.

 It has been released on DVD.
</item 3>

<item 4>
Sam Eagle
Sam Eagle is a Muppet character originating from the television show "The Muppet Show", where he was performed by Frank Oz.

 Sam has appeared in every Muppet film; as himself in "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "Muppets from Space", and "The
Muppets", as well as the Head Schoolmaster in "The Muppet Christmas Carol," Samuel Arrow in "Muppet Treasure Island" and a CIA agent in "Muppets Most Wanted."

 He also appears in the television series, "The Muppets."
</item 4>

<item 5>
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 5>

<item 6>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 6>

<item 7>
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 7>

<item 8>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 8>

<item 9>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 9>

<item 10>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 10>
=== END ITEM id=ecae4153215468e7 ===

=== ITEM id=51e4224a81ac4488 ===
Gold sentences and facts:
Sentence 0:  It has its offices in New Roc City in New Rochelle, New York.
  Fact 0.0: An entity has its offices in New Roc City in New Rochelle, New York.
Sentence 1: New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in Westchester County, New York.
  Fact 1.0: New Roc City is also known as New Rochelle Center.
  Fact 1.1: New Roc City is an entertainment complex.
  Fact 1.2: New Roc City is a retail complex.
  Fact 1.3: New Roc City is a residential complex.
  Fact 1.4: New Roc City is in the Downtown section of the city of New Rochelle.
  Fact 1.5: New Rochelle is in Westchester County, New York.

Retrieved items:
<item 1>
The entries cover notable U.S. buildings, districts, businesses, and people. New York features prominently: office towers such as 888 Seventh Avenue, 245 Park
Avenue, 383 and 270 Madison/Park Avenue, and Goldman Sachs’ 200 West Street; landmarks including the Woolworth Building and 90 West Street; and residential or
mixed-use towers such as MiMA and Random House Tower. Other topics include historic theaters, retail and media companies, transportation firms, and real-estate
developer
</item 1>

<item 2>
Japanese Weekend School of New York
The Japanese Weekend School of New York (JWSNY; ニューヨーク "Nyūyōku Hoshū Jugyō Kō") is a Japanese supplementary school in the New York City metropolitan area.

 It has its offices in New Roc City in New Rochelle, New York.

 The Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi Kai") manages the school system, and the JWSNY is one of
its two weekend school systems.

 The JEI also operates two Japanese day schools in the New York area.
</item 2>

<item 3>
Church of St. Ignatius Loyola (New York City)
Ignatius's School on the north midblock location of 84th Street behind the church and the high school of Loyola School (also 980 Park Avenue) at the northwest
corner of Park Avenue and 83rd Street.

 In addition, another Jesuit high school, Regis High School (55 E 84th Street), occupies the midblock location on the north side of 84th Street.

 The church was added to the National Register of Historic Places on July 24, 1980.
</item 3>

<item 4>
New Rochelle Walk of Fame
The New Rochelle Walk of Fame was installed in 2011 in Ruby Dee Park at Library Green, located in the downtown section of the city of New Rochelle in
Westchester County, New York.

 The "walk" is a tribute to some of New Rochelle's most notable residents from throughout its 325-year history.

 It was created and funded by former resident Roderick Kennedy, Jr., working in partnership with the City of New Rochelle and the New Rochelle Business
Improvement District.
</item 4>

<item 5>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
</item 5>

<item 6>
New Roc City
New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in
Westchester County, New York.

 It is located at 33 LeCount Place, between Main Street North and Main Street South.

 The center was built on the site of the former New Rochelle Mall which closed in 1992.
</item 6>

<item 7>
New Rochelle Harbor (Long Island Sound)
New Rochelle Harbor is the name of a harbor located along Long Island Sound in the city of New Rochelle in Westchester County, New York.

 The Davenport Neck peninsula off the mainland divides New Rochelle's waterfront into two bays; the westerly referred to as New Rochelle Harbor and the easterly
as Echo Bay.
</item 7>

<item 8>
Montefiore New Rochelle Hospital
Montefiore New Rochelle Hospital (formerly Sound Shore Medical Center) is a community-based, teaching hospital located at 16 Guion Place in the city of New
Rochelle, in Westchester County, New York, and affiliated with the Albert Einstein College of Medicine.

 The hospital opened in 1892 as New Rochelle Hospital.

 On November 6, 2013, Sound Shore was acquired by the Bronx-based Montefiore Medical Center, the University Hospital for the Albert Einstein College of
Medicine, was renamed Montefiore New Rochelle Hospital, and became part of the Montefiore Health System.
</item 8>

<item 9>
Standard Star Building
The Standard Star Building is a historic commercial structure located in the Downtown section of New Rochelle, Westchester County, New York.

 The building, designed by architect Lawrence J. Barnard, was completed in 1924 and is an architecturally significant example of the Italian Renaissance style
in New Rochelle.

 Although it has been altered, these changes occurred only on the Le Count Place façade and the interior of the building.

 The original Standard Star building exterior remains largely unchanged.

 It is further historically significant for its association with a long-published New Rochelle newspaper covering life in New Rochelle.
</item 9>

<item 10>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 10>
=== END ITEM id=51e4224a81ac4488 ===

=== ITEM id=68ab9b2cee46b9cc ===
Gold sentences and facts:
Sentence 0: Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock, and jazz.
  Fact 0.0: Stephen Lawrence 'Steve' Winwood was born 12 May 1948.
  Fact 0.1: Steve Winwood is an English musician.
  Fact 0.2: Steve Winwood's genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock, and jazz.
Sentence 1: Per "Pelle" Almqvist (born 29 May 1978), also known as Howlin' Pelle Almqvist, is a Swedish singer and songwriter.
  Fact 1.0: Per 'Pelle' Almqvist was born 29 May 1978.
  Fact 1.1: Per 'Pelle' Almqvist is also known as Howlin' Pelle Almqvist.
  Fact 1.2: Per 'Pelle' Almqvist is a Swedish singer and songwriter.

Retrieved items:
<item 1>
Steve Winwood
Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock,
and jazz.

 Though primarily a vocalist and keyboardist, Winwood also plays bass guitar, drums, acoustic and electric guitar, mandolin, violin, and other strings.
</item 1>

<item 2>
Steve Alvord
Steven Lee Alvord (born October 2, 1964 in Bellingham, Washington) is a former professional American football defensive tackle and defensive end in the National
Football League and World League of American Football.

 In his four-year pro career he played for the St. Louis/Phoenix Cardinals of the NFL, and the Barcelona Dragons of the WLAF.

 Alvord played college football at Washington.

He has a cool daughter, happy birthday Katie.
</item 2>

<item 3>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 3>

<item 4>
Graph query row:
person: Per "Pelle" Almqvist
birth_date: 29 May 1978
</item 4>

<item 5>
Steve Winwood (album)
Steve Winwood is the debut solo studio album by blue-eyed soulster Steve Winwood.

 It was released three years after the break-up of Traffic.

 Though the album sold moderately well in the USA, it was a commercial disappointment compared to Traffic's recent albums, peaking at number 22 on the
"Billboard" 200 album chart.

 In Winwood's home country the situation was reversed; while Traffic's recent albums had only been moderately successful in the UK, "Steve Winwood" reached
number 12 on The Official Charts.

 Island Records launched two singles from the album, "Hold On" and "Time Is Running Out", both of which failed to make the charts.
</item 5>

<item 6>
Pelle Almqvist
Per "Pelle" Almqvist (born 29 May 1978), also known as Howlin' Pelle Almqvist, is a Swedish singer and songwriter.

 He is the lead singer of garage rock band The Hives.
</item 6>

<item 7>
Glen Matlock
Glen Matlock (born 27 August 1956) is an English musician best known for being the bass guitarist in the original line-up of the punk rock band the Sex Pistols.

 He is sometimes credited as a co-author on 10 of the 12 songs on "Never Mind the Bollocks, Here's the Sex Pistols", although he had left the band by the time
the record was released and the bass was played by Steve Jones.

 He also continues to make his own records and tour with various bands, including the Sex Pistols.
</item 7>

<item 8>
Revolutions – The Very Best of Steve Winwood
Revolutions – The Very Best of Steve Winwood is the sixth compilation album by Steve Winwood.

 The album includes music from Winwood's solo career, as well as groups with which he has performed, including The Spencer Davis Group, Traffic, and Blind
Faith.

 The CD was released as a box set and a single disc.

 As of October 2014, the box set is out of print (but can be purchased digitally), while the single disc is still available.

 The songs "The Finer Things" and "Roll With It" are exclusive to the single disc version and cannot be found on the box set.
</item 8>

<item 9>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 9>

<item 10>
Graph query row:
person: Steve Winwood
birth_date: 12 May 1948
</item 10>
=== END ITEM id=68ab9b2cee46b9cc ===

=== ITEM id=2be0947f374d1636 ===
Gold sentences and facts:
Sentence 0: Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to become lead singer of the hard rock supergroup "Rock Star Supernova".
  Fact 0.0: Lukas Rossi was born on December 21, 1976.
  Fact 0.1: Lukas Rossi is a Canadian rock musician.
  Fact 0.2: Lukas Rossi was the winner of a CBS Television reality series.
  Fact 0.3: That CBS Television reality series was a televised audition contest to become lead singer of the hard rock supergroup 'Rock Star Supernova'.
Sentence 1: Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and founding members of the rock band Chicago.
  Fact 1.0: Terry Alan Kath was born on January 31, 1946.
  Fact 1.1: Terry Alan Kath died on January 23, 1978.
  Fact 1.2: Terry Alan Kath was an American musician.
  Fact 1.3: Terry Alan Kath was an American songwriter.
  Fact 1.4: Terry Alan Kath was best known as the original guitarist of the rock band Chicago.
  Fact 1.5: Terry Alan Kath was one of the lead singers of the rock band Chicago.
  Fact 1.6: Terry Alan Kath was a founding member of the rock band Chicago.

Retrieved items:
<item 1>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 1>

<item 2>
Roger Kitter
Roger Kitter (20 October 1949 – 3 January 2015) was an English actor best known for playing Captain Alberto Bertorelli in series 7 of the British sitcom TV
series "'Allo 'Allo!

".

 He had previously appeared weekly with Lulu throughout the 10-week run of her 1973 BBC1 series "It's Lulu".

 With Kaplan Kaye he also recorded a song "Chalk Dust – the Umpire Strikes Back" using the moniker 'The Brat'.

 Released on the Hansa label, it entered the UK Singles Chart on 10 July 1982; it reached a peak of number 19, and remained in the chart for 8 weeks.

 The song was a Top 10 hit in the Netherlands, Belgium and South Africa, and lampooned John McEnroe complaining about line calls in tennis ("The ball's in,
everyone can see that the ball's in!")

.

 He was also an impressionist, who provided the voice of Tommy Cooper in the Lego 'Kipper' advertisement.
</item 2>

<item 3>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 3>

<item 4>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 4>

<item 5>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 5>

<item 6>
Headspin (song)
"Headspin" is a song by unsigned Canadian band Rise Electric, which is fronted by "" winner Lukas Rossi.

 The song was released on Rock Star Supernova's self-titled debut album.

 It was released as the third single from the album in early 2007.

 Rossi performed "Headspin" three times while on "Rock Star: Supernova".

 Following the Rock Star Supernova tour Luke Rossi released an acoustic version of the song which is featured on his album "Love & Lust".
</item 6>

<item 7>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 7>

<item 8>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 8>

<item 9>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 9>

<item 10>
The Halo Method
The Halo Method Is an American rock supergroup, formed in 2012 in Hollywood, California.

 It consisted of guitarist Ben Moody (We Are the Fallen, ex-Evanescence), drummer Miles McPherson, vocalist Lukas Rossi (ex-Rock Star Supernova, ex-Daylight
Division) and bassist Josh Newell (ex-In This Moment).
</item 10>
=== END ITEM id=2be0947f374d1636 ===
