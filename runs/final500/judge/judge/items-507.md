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

=== ITEM id=629e5e8a9991ffa6 ===
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
The Apple Dumpling Gang (film)
The Apple Dumpling Gang is a 1975 American comedy-western film produced by Walt Disney Productions about a slick gambler named Russell Donovan (Bill Bixby) who
is duped into taking care of a group of orphans who eventually strike gold during the California Gold Rush.
</item 1>

<item 2>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 2>

<item 3>
The Apple Dumpling Gang
The Apple Dumpling Gang is a 1971 novel by Jack Bickham, about a group of orphaned children during the California gold rush.

 They encounter a gambler who reluctantly helps them, as well as a pair of hapless robbers who are after the gold the children have found.
</item 3>

<item 4>
The Sasquatch Gang
The Sasquatch Gang (also known as The Sasquatch Dumpling Gang) is a 2006 comedy film written and directed by Tim Skousen, the first assistant director on
"Napoleon Dynamite".
</item 4>

<item 5>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 5>

<item 6>
Heavyweights
Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.

 The film centers around a fat camp for kids that is taken over by a fitness guru named Tony Perkis (Ben Stiller).
</item 6>

<item 7>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 7>

<item 8>
Gun Shy (TV series)
Gun Shy is an American sitcom that was shown on CBS from March 15 to April 19, 1983.

 The series, produced by Walt Disney Productions, was based on its popular comedy-western films: "The Apple Dumpling Gang" and "The Apple Dumpling Gang Rides
Again".
</item 8>

<item 9>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 9>

<item 10>
Buddy Baker (composer)
Norman Dale "Buddy" Baker (January 4, 1918 – July 26, 2002) was an American composer who, together with Paul J. Smith, scored many Disney films, such as "The
Apple Dumpling Gang" in 1975, "The Apple Dumpling Gang Rides Again" in 1979, "The Shaggy D.A." in 1976, "The Many Adventures of Winnie the Pooh" in 1977, and
"The Fox and the Hound" in 1981.
</item 10>
=== END ITEM id=629e5e8a9991ffa6 ===

=== ITEM id=a1cf3e938a467dfb ===
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
Grand Central Parkway
The Grand Central Parkway (GCP) is a 14.61-mile (23.51 km) long parkway that stretches from the Triborough Bridge in New York City to Nassau County on Long
Island.

 At the Queens–Nassau border, it becomes the Northern State Parkway, which runs across the northern part of Long Island through Nassau County and into Suffolk
County, where it ends in Hauppauge.

 The westernmost stretch (from the Triborough Bridge to exit 4) also carries a short stretch of Interstate 278 (I-278).

 The parkway runs through Queens and passes the Cross Island Parkway, Long Island Expressway, LaGuardia Airport and Citi Field, home of the New York Mets.

 The parkway is designated New York State Route 907M (NY 907M), an unsigned reference route.

 Despite its name, the Grand Central Parkway was not named after Grand Central Terminal.
</item 1>

<item 2>
Saddle Rock Elementary School
Saddle Rock Elementary School or simply "Saddle Rock" is an elementary school, comprising grades Kindergarten through 5.

 It is a public school located in Great Neck, New York, USA as part of the Great Neck School District.

 The school has been recognized by the Blue Ribbon Schools Program.
</item 2>

<item 3>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
</item 3>

<item 4>
Village School (Great Neck, New York)
Village School (also known as VS) is an American long-established public alternative school.

 It is a member of Coalition of Essential Schools and is located in the village of Great Neck, New York, serving students in grades 8 through 12.

 Village School is one of three high schools in the Great Neck School District, which includes Great Neck North High School and Great Neck South High School.

 Village School offers its 39 students an outdoor education program, college preparatory program, and inclusion of students with disabilities.
</item 4>

<item 5>
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
</item 5>

<item 6>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 6>

<item 7>
Cayuga Secondary School
Cayuga Secondary School is a secondary school located at 70 Highway 54, Cayuga, Ontario, Canada.

 It is part of the Grand Erie District School Board.

 Cayuga Secondary School opened in 1963 under the name Cayuga Technical and Commercial High School since it did not offer a grade 13 program.

 It earned secondary school status in 1970.

 The students come from J. L. Mitchener, Rainham Central, Seneca Central Seneca Unity and Oneida Central public schools, Caledonia as well as some students from
the Six Nations Reserve and from the Catholic elementary school St. Stephen's.

 The school currently has about 600 students enrolled.

 It is the home of the Warriors Football Team, The team has currently joined "Forces" With one of its two rivals(Hagersville/McKinnon) Hagersville.

 After a previous coach left after the 2015 season the two schools contacted one another and became the "Warricanes"
</item 7>

<item 8>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
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
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 10>
=== END ITEM id=a1cf3e938a467dfb ===

=== ITEM id=4fe3926eaf0ad0fd ===
Gold sentences and facts:
Sentence 0: Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal Navy surface vessels and submarines.
  Fact 0.0: Rosyth Dockyard is a large naval dockyard
  Fact 0.1: Rosyth Dockyard is on the Firth of Forth
  Fact 0.2: Rosyth Dockyard is at Rosyth, Fife, Scotland
  Fact 0.3: Rosyth Dockyard is owned by Babcock Marine
  Fact 0.4: Rosyth Dockyard formerly undertook refitting of Royal Navy surface vessels
  Fact 0.5: Rosyth Dockyard formerly undertook refitting of Royal Navy submarines
Sentence 1: The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.
  Fact 1.0: The Queen Elizabeth class is a class of two aircraft carriers
  Fact 1.1: The Queen Elizabeth class aircraft carriers belong to the United Kingdom's Royal Navy

Retrieved items:
<item 1>
St Vincent-class battleship
The "St Vincent"-class battleships were a group of three dreadnought battleships built for the Royal Navy in the first decade of the 20th century.

 The sister ships spent their entire careers assigned to the Home and Grand Fleets. Aside from participating in the Battle of Jutland in May 1916 and the
inconclusive Action of 19 August several months later, their service during the First World War generally consisted of routine patrols and training in the North
Sea.

 "Vanguard" was destroyed in 1917 by a magazine explosion with the near total loss of her crew.

 The remaining pair were obsolete by the end of the war in 1918, and spent their remaining time either in reserve or as training ships before being sold for
scrap in the early 1920s.
</item 1>

<item 2>
Queen Elizabeth-class aircraft carrier
The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.

 The first,  , was named on 4 July 2014, with her ship commissioning planned for 2017, and an initial operating capability expected in 2018.

 The second, HMS "Prince of Wales" , is scheduled to be launched around Summer 2017, followed by commissioning in 2020 and service thereafter.

 On 5 September 2014, at the NATO 2014 Wales summit, the Prime Minister David Cameron announced that the second carrier would be brought into service, ending
years of uncertainty surrounding its future.

 This was confirmed by the November 2015 Government Strategic Defence Review, with both carriers entering service, one being available at any time.
</item 2>

<item 3>
Goliath (Rosyth)
Goliath is a crane in Rosyth Dockyard, Scotland, with a lift capacity of 1000 t , the largest in Britain.
</item 3>

<item 4>
British Rail Class 159
The Class 159 is a class of British diesel multiple-unit trains of the Sprinter family, built in 1989–1992 by British Rail Engineering Limited (BREL)'s Derby
Litchurch Lane Works as Class 158.

 Before entering traffic, the original 22 units were modified at Rosyth Dockyard to Class 159 to operate express services from London Waterloo to Exeter and
replace locomotive-hauled passenger trains.
</item 4>

<item 5>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 5>

<item 6>
Rosyth Dockyard
Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal
Navy surface vessels and submarines.

 Before its privatisation in the 1990s it was formally the Royal Naval Dockyard Rosyth.

 Its primary role is now as integration site for the Royal Navy's newest aircraft carriers - the "Queen Elizabeth"-class.
</item 6>

<item 7>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 7>

<item 8>
Rosyth railway station
Rosyth railway station serves the town of Rosyth in Fife, Scotland.

 The station is managed by Abellio ScotRail and lies on the Fife Circle Line, 23.6 km north of Edinburgh Waverley .

 It was opened in 1917 by the North British Railway (as "Rosyth Halt") to serve the nearby naval dockyard.
</item 8>

<item 9>
HMS Prince of Wales (R09)
HMS "Prince of Wales" is the second "Queen Elizabeth"-class aircraft carrier under construction for the Royal Navy, with plans for active service from 2020.

 She is the seventh Royal Navy ship to have the name HMS "Prince of Wales".

 Construction of the ship began in 2011 at Rosyth Dockyard and in April 2016, was said to be around 80% structurally complete.
</item 9>

<item 10>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 10>
=== END ITEM id=4fe3926eaf0ad0fd ===

=== ITEM id=cdb914ed738d48f4 ===
Gold sentences and facts:
Sentence 0: Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.
  Fact 0.0: Rick Ray is an American filmmaker.
  Fact 0.1: Rick Ray is best known for his 2006 documentary film 10 Questions for the Dalai Lama.
  Fact 0.2: Rick Ray wrote, filmed and directed 10 Questions for the Dalai Lama.
Sentence 1: John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.
  Fact 1.0: John Alan Lasseter was born January 12, 1957.
  Fact 1.1: John Alan Lasseter is an American animator.
  Fact 1.2: John Alan Lasseter is an American film director.
  Fact 1.3: John Alan Lasseter is an American screenwriter.
  Fact 1.4: John Alan Lasseter is an American film producer.
Sentence 2:  He is also the Principal Creative Advisor for Walt Disney Imagineering.
  Fact 2.0: John Alan Lasseter is also the Principal Creative Advisor for Walt Disney Imagineering.

Retrieved items:
<item 1>
Wreck-It Ralph
Wreck-It Ralph is a 2012 American 3D computer-animated fantasy-comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 52nd Disney animated feature film.

 The film was directed by Rich Moore, who has directed episodes of "The Simpsons" and "Futurama", and the screenplay was written by Phil Johnston and Jennifer
Lee from a story by Moore, Johnston, and Jim Reardon.

 John Lasseter served as the executive producer.

 The film features the voices of John C. Reilly, Sarah Silverman, Jack McBrayer, and Jane Lynch.

 The film tells the story of the eponymous arcade game villain who rebels against his role and dreams of becoming a hero.

 He travels between games in the arcade and ultimately must eliminate a dire threat that could affect the entire arcade and one that Ralph himself
unintentionally started.
</item 1>

<item 2>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 2>

<item 3>
John Lasseter (born 1957) is an American animator, director, writer, and producer who led creative work at Pixar, Disney Animation, and DisneyToon, and advised
Walt Disney Imagineering. He directed Pixar’s pioneering shorts *Luxo Jr.* and *Red’s Dream*, and made his feature debut with 1995’s *Toy Story*. He
co-developed the *Toy Story* story and directed *Cars 2*; other listed films include *Toy Story
</item 3>

<item 4>
Lasseter Family Winery
Lasseter Family Winery is a winery located in Glen Ellen, Sonoma County, California.

 The winery was founded in 2000 by Pixar and Walt Disney Animation Studios CCO John Lasseter and his wife, Nancy Lasseter.

 The winery, once inhabited by the Grand Cru Winery, produces approximately 1,200 cases of French red wine blends annually, with the capacity to produce up to
6,000.

 The winery grows Bordeaux and Rhône varietals on 27 acres.

 One of the Lasseters' winemaking mentors was Jess Jackson, of Kendall-Jackson.
</item 4>

<item 5>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 5>

<item 6>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 6>

<item 7>
Red's Dream
Red's Dream is a 1987 American computer-animated short film produced by Pixar and directed by John Lasseter.

 The short film, which runs four minutes, stars Red, a unicycle.

 Propped up in the corner of a bicycle store on a rainy night, Red dreams about a better place.

 "Red's Dream" was Pixar's second computer-animated short following "Luxo Jr." in 1986, also directed by Lasseter.
</item 7>

<item 8>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 8>

<item 9>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 9>

<item 10>
Walt Disney Pictures, founded under that name in 1983, produces live-action films; Walt Disney Studios Motion Pictures distributes Disney studio releases,
succeeding Buena Vista (1953–2007). Disney highlights include *Bambi* (1942), *Pinocchio* (1940), and the 1989–99 Renaissance, spanning ten animated films from
*The Little Mermaid* to *Tarzan*. Buddy Baker scored several Disney films. Other entries cover Mickey and Donald, the
</item 10>
=== END ITEM id=cdb914ed738d48f4 ===

=== ITEM id=cb4552df7ff65b81 ===
Gold sentences and facts:
Sentence 0: Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.
  Fact 0.0: Christopher Edward Nolan was born 30 July 1970
  Fact 0.1: Christopher Edward Nolan is English-American
  Fact 0.2: Christopher Edward Nolan is a film director
  Fact 0.3: Christopher Edward Nolan is a producer
  Fact 0.4: Christopher Edward Nolan is a screenwriter
Sentence 1: Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.
  Fact 1.0: Paul Joseph Schrader was born July 22, 1946
  Fact 1.1: Paul Joseph Schrader is an American screenwriter
  Fact 1.2: Paul Joseph Schrader is a film director
  Fact 1.3: Paul Joseph Schrader is a film critic

Retrieved items:
<item 1>
Ian Schneider
Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
</item 1>

<item 2>
The Walker
The Walker is a 2007 American-British drama film written and directed by Paul Schrader.

 It is an independent production and is the latest installment in Schrader's "night workers" series of films, starting with "Taxi Driver" in 1976, followed by
"American Gigolo" in 1980 and "Light Sleeper" in 1992.
</item 2>

<item 3>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 3>

<item 4>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 4>

<item 5>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 5>

<item 6>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 6>

<item 7>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
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
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 9>

<item 10>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 10>
=== END ITEM id=cb4552df7ff65b81 ===

=== ITEM id=f58b7f04e2a4f17c ===
Gold sentences and facts:
Sentence 0: Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
  Fact 0.0: Milium's vernacular name is milletgrass.
  Fact 0.1: Milium is a genus of plants in the grass family.
  Fact 0.2: Milium plants are Eurasian, North American, and North African.
Sentence 1: Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.
  Fact 1.0: Eucommia is a genus of small trees.
  Fact 1.1: Eucommia trees are now native to China.
  Fact 1.2: The fossil record of Eucommia shows a much wider distribution.
Sentence 2:  The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in herbology such as traditional Chinese medicine.
  Fact 2.0: Eucommia has a single living species, Eucommia ulmoides.
  Fact 2.1: Eucommia ulmoides is near threatened in the wild.
  Fact 2.2: Eucommia ulmoides is widely cultivated in China for its bark.
  Fact 2.3: Eucommia ulmoides is highly valued in herbology, such as traditional Chinese medicine.

Retrieved items:
<item 1>
Milium (plant)
Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
</item 1>

<item 2>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 2>

<item 3>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 3>

<item 4>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 4>

<item 5>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 5>

<item 6>
Ocimum
Ocimum is a genus of aromatic annual and perennial herbs and shrubs in the family Lamiaceae, native to the tropical and warm temperate regions of all 6
inhabited continents, with the greatest number of species in Africa.

 Its best known species are the cooking herb Cooking basil, "O. basilicum" and the medicinal herb Tulsi (holy basil), "O. tenuiflorum".
</item 6>

<item 7>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 7>

<item 8>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 8>

<item 9>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 9>

<item 10>
Colocasia
Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.

 Some species are widely cultivated and naturalized in other tropical and subtropical regions.

 Common names include tarul, karkala ko ganu, elephant-ear, taro, cocoyam, dasheen, chembu, champadhumpa, shavige gadde, and eddoe.

 Elephant-ear and cocoyam are also used for some other large-leaved genera in the Araceae, notably "Xanthosoma" and "Caladium".

 The generic name is derived from the ancient Greek word "kolokasion", which in the Greek botanist Dioscorides (1st century AD) meant the edible roots of both
"Colocasia esculenta" and "Nelumbo nucifera".

 It is thought that the edible roots of "Colocasia esculenta" have been cultivated in Asia for more than ten thousand years.

 The species "Colocasia esculenta" is an invasive species in wetlands along the American Gulf coast, where it threatens to displace native wetland plants.
</item 10>
=== END ITEM id=f58b7f04e2a4f17c ===

=== ITEM id=70db1814d75e7481 ===
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
The entries profile actors and television productions. Tommy Knight is known for The Sarah Jane Adventures and Doctor Who; Mark Benton played Daniel Chalk in
Waterloo Road; Robin Chalk appeared in Dirty Dancing and Moon. Tamzin Outhwaite starred in EastEnders, Red Cap, Paradox, Vital Signs and Inside No. 9’s “Empty
Orchestra.” Dominic West is known for The Wire and Appropriate Adult; John Altman played Nick Cotton in EastEnders. Other entries cover children’s
</item 1>

<item 2>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 2>

<item 3>
The entries profile actors, voice performers, puppeteers, composers, writers, and animation history. Ken Barrie voiced and sang for *Postman Pat*; Roger Kitter
played Captain Bertorelli in *’Allo ’Allo!* and scored a tennis-themed chart hit. Ralph Bakshi directed influential adult animations including *Fritz the Cat*,
*Heavy Traffic*, and *Fire and Ice*, plus live-action *Cool and the Crazy*. Other profiles include Disney composer Buddy Baker; Sc
</item 3>

<item 4>
The Church Mouse
The Church Mouse is a 1934 British comedy film directed by Monty Banks and starring Laura La Plante, Ian Hunter and Edward Chapman.

 It was made by the British subsidiary of Warner Brothers at the company's Teddington Studios.

 It was made as a more expensive production than much of the studio's low-budget quota quickie output.
</item 4>

<item 5>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 5>

<item 6>
Keep Your Seats, Please
Keep Your Seats, Please is a 1936 British comedy film directed by Monty Banks and starring George Formby, Florence Desmond and Alastair Sim.

 It marked the film debut of the child star Binkie Stuart.

 The film was made by Associated Talking Pictures.
</item 6>

<item 7>
Monty Banks
Montague (Monty) Banks (15 July 1897 [registered on 18 July 1897] – 7 January 1950 born Mario Bianchi) was an Italian comedian and film director who achieved
success in the United States and in England.
</item 7>

<item 8>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 8>

<item 9>
Mark Benton
Mark Benton (born 16 November 1965) is an English actor, known for his roles as Eddie in "Early Doors", Howard in "Northern Lights" and Martin Pond in
"Barbara".

 Benton has also starred in BBC One drama, "Waterloo Road" as maths teacher, Daniel Chalk from 2011 to 2013 and again in 2014.
</item 9>

<item 10>
George Seaton
George Seaton (April 17, 1911 – July 28, 1979) was an American screenwriter, playwright, film director and producer, and theatre director.
</item 10>
=== END ITEM id=70db1814d75e7481 ===

=== ITEM id=3c18678a43273010 ===
Gold sentences and facts:
Sentence 0:  They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.
  Fact 0.0: This team is most well known for including Sugar Rashad Evans.
  Fact 0.1: Sugar Rashad Evans is a former UFC light heavyweight champion.
  Fact 0.2: This team is most well known for including Vitor Belfort.
  Fact 0.3: Vitor Belfort is a former UFC light heavyweight champion.
  Fact 0.4: This team is most well known for including Eddie Alvarez.
  Fact 0.5: Eddie Alvarez is a former UFC lightweight champion.
Sentence 1: Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.
  Fact 1.0: Vitor Vieira Belfort was born on April 1, 1977.
  Fact 1.1: Vitor Vieira Belfort is a Brazilian mixed martial artist.
  Fact 1.2: Vitor Vieira Belfort is currently competing in the Middleweight division of the Ultimate Fighting Championship.

Retrieved items:
<item 1>
Rashad Evans
Rashad Anton Evans (born September 25, 1979) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2003, Evans was the Heavyweight winner of The Ultimate Fighter 2 and is also a former UFC Light Heavyweight Champion.
</item 1>

<item 2>
Paulo Borrachinha
Paulo Henrique Costa (born April 21, 1991), better known as Paulo Borrachinha, is a Brazilian mixed martial artist currently competing in the Middleweight
division of the Ultimate Fighting Championship.

 In the day August 3 of 2017, Borracinha and the Clube Atlético Mineiro officialized institutional partnership.

 The agreement did not create a bond between the club and the athlete.

 The agreement allows the use of the athletic shield of Atlético in the training uniforms of the fighter, within the possibilities of his contract with the UFC.
</item 2>

<item 3>
Jack Marshman
Jack "Hammer" Marshman (born December 19, 1989) is a Welsh mixed martial artist.

 He was the Middleweight Cage Warriors Fighting Championship (CWFC) champion and Lonsdale Middleweight Champion.

 He currently competes in Middleweight division in the Ultimate Fighting Championship(UFC).
</item 3>

<item 4>
Kelvin Gastelum
Kelvin Gastelum (born October 24, 1991) is an American professional mixed martial artist who is currently signed to the Ultimate Fighting Championship (UFC).

 He was the winner of "The Ultimate Fighter 17".

 Gastelum is currently competing as a middleweight and is ranked #8 in the UFC official middleweight rankings.
</item 4>

<item 5>
Luke Zachrich
Luke Zachrich (born October 1, 1981) is an American mixed martial artist and former professional boxer who formerly competed in the Middleweight division of the
Ultimate Fighting Championship A professional mixed martial arts competitor since 2006, Zachrich was a member of Team Forrest on the seventh season of The
Ultimate Fighter, and has also fought for King of the Cage, Xtreme Fighting Organization, Bellator, and Ultimate Victory Challenge, the latter of which he was
the inaugural Middleweight Champion.
</item 5>

<item 6>
David Branch (fighter)
David Branch (born September 26, 1981) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2007, he has also formerly competed for the World Series of Fighting, Bellator, Shark Fights, and Titan FC.

 Branch was the inaugural World Series of Fighting Light Heavyweight and Middleweight Champion.

 He is currently ranked as the #9 Middleweight in the world according to Sherdog.
</item 6>

<item 7>
Vitor Belfort
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 He is the former UFC Light Heavyweight Champion, UFC 12 Heavyweight Tournament Champion, and the Cage Rage World Light Heavyweight Champion.

 As of June 16, 2017, he is the #11 contender in official UFC Middleweight rankings.
</item 7>

<item 8>
Sam Alvey
Samuel Alvey (born May 6, 1986) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.

 A professional MMA competitor since 2008, Alvey has also formerly competed for Bellator, King of the Cage, the MFC, and was a contestant on .

 As of March 2017, Alvey is #14 in the official UFC Middleweight Rankings.
</item 8>

<item 9>
Blackzilians
The Blackzilians is a defunct professional team of fighters in mixed martial arts, boxing, kickboxing, amateur wrestling and catch wrestling located in South
Florida.

 They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC
lightweight champion Eddie Alvarez.
</item 9>

<item 10>
Luke Barnatt
Luke Barnatt (born 13 April 1988) is an English mixed martial artist who fought as a Middleweight for the Ultimate Fighting Championship.

 He was a member of FX's , and has also competed for BAMMA and UCMMA in England.

 At 6' 6" Barnatt is one of the tallest fighters currently competing in his weight class.

 Barnatt received his nickname "Bigslow" from brothers John Maguire and Tommy Maguire, for his size and being slow in response to their jokes and insults.

 Barnatt competes in the middleweight division of the Absolute Championship Berkut.
</item 10>
=== END ITEM id=3c18678a43273010 ===

=== ITEM id=a7880dc98bd4b6c4 ===
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
Chicago’s “25 or 6 to 4,” written by Robert Lamm and sung by Peter Cetera, was recorded in 1969 and became the band’s first U.S. top-five hit; Terry Kath plays
its wah-wah guitar solo. Kath, Chicago’s founding guitarist, singer, and songwriter, also leads “Free” and “Brand New Love Affair” (first half) and sings
“Wishing You Were Here.” His final album was 1977’s *
</item 1>

<item 2>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 2>

<item 3>
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
</item 3>

<item 4>
Steve Augeri is an American rock singer known for Tall Stories, Tyketto, and Journey. English multi-instrumentalist Steve Winwood performed with the Spencer
Davis Group, Traffic, and Blind Faith; his debut solo album reached UK No. 12, while two compilations document his work. His 1986 hit “Back in the High Life
Again” topped the US Adult Contemporary chart for three weeks. Pedal-steel guitarist Speedy West played with Jimmy Bryant and on recordings by Tennessee
</item 4>

<item 5>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 5>

<item 6>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 6>

<item 7>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 7>

<item 8>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 8>

<item 9>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 9>

<item 10>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 10>
=== END ITEM id=a7880dc98bd4b6c4 ===

=== ITEM id=d134260b6d3b884f ===
Gold sentences and facts:
Sentence 0: Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
  Fact 0.0: Robert Paul Holdstock was born on 2 August 1948.
  Fact 0.1: Robert Paul Holdstock died on 29 November 2009.
  Fact 0.2: Robert Paul Holdstock was an English novelist.
  Fact 0.3: Robert Paul Holdstock was an English author.
  Fact 0.4: Robert Paul Holdstock was best known for his works of Celtic fantasy literature.
  Fact 0.5: Robert Paul Holdstock was best known for his works of Nordic fantasy literature.
  Fact 0.6: Robert Paul Holdstock was best known for his works of Gothic fantasy literature.
  Fact 0.7: Robert Paul Holdstock was best known for his works of Pictish fantasy literature.
  Fact 0.8: Robert Paul Holdstock's works are predominantly in the fantasy subgenre of mythic fiction.
Sentence 1: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 1.0: Henry Louis Mencken was born on September 12, 1880.
  Fact 1.1: Henry Louis Mencken died on January 29, 1956.
  Fact 1.2: Henry Louis Mencken was an American journalist.
  Fact 1.3: Henry Louis Mencken was an American satirist.
  Fact 1.4: Henry Louis Mencken was an American cultural critic.
  Fact 1.5: Henry Louis Mencken was a scholar of American English.

Retrieved items:
<item 1>
Robert Holdstock bibliography
This is a bibliography of fantasy author Robert Holdstock.
</item 1>

<item 2>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 2>

<item 3>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 3>

<item 4>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 4>

<item 5>
Robert Holdstock
Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish
fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
</item 5>

<item 6>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 6>

<item 7>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 7>

<item 8>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 8>

<item 9>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 9>

<item 10>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 10>
=== END ITEM id=d134260b6d3b884f ===

=== ITEM id=c09389abac28b88b ===
Gold sentences and facts:
Sentence 0: The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
  Fact 0.0: The Second Battle of Guam was fought from 21 July to 10 August 1944.
  Fact 0.1: The Second Battle of Guam was the American recapture of the Japanese-held island of Guam.
  Fact 0.2: Guam is a U.S. territory in the Mariana Islands.
  Fact 0.3: Japan captured Guam from the U.S. in the 1941 First Battle of Guam.
  Fact 0.4: The Second Battle of Guam took place during the Pacific campaign of World War II.
Sentence 1: The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and Antietam Creek as part of the Maryland Campaign.
  Fact 1.0: The Battle of Antietam was also known as the Battle of Sharpsburg, particularly in the South.
  Fact 1.1: The Battle of Antietam was fought on September 17, 1862.
  Fact 1.2: The Battle of Antietam was fought near Sharpsburg, Maryland.
  Fact 1.3: The Battle of Antietam was fought near Antietam Creek.
  Fact 1.4: The Battle of Antietam was part of the Maryland Campaign.

Retrieved items:
<item 1>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 1>

<item 2>
Maryland Campaign
The Maryland Campaign—or Antietam Campaign—occurred September 4–20, 1862, during the American Civil War.

 Confederate Gen. Robert E. Lee's first invasion of the North was repulsed by the Army of the Potomac under Maj. Gen. George B. McClellan, who moved to
intercept Lee and his Army of Northern Virginia and eventually attacked it near Sharpsburg, Maryland.

 The resulting Battle of Antietam was the bloodiest single-day battle in American history and is widely considered one of the major turning points of the war.
</item 2>

<item 3>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 3>

<item 4>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 4>

<item 5>
Battle of Antietam
The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and
Antietam Creek as part of the Maryland Campaign.

 It was the first field army–level engagement in the Eastern Theater of the American Civil War to take place on Union soil and is the bloodiest single-day
battle in American history, with a combined tally of 22,717 dead, wounded, or missing.
</item 5>

<item 6>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 6>

<item 7>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 7>

<item 8>
Graph query row:
event: Battle of Antietam
date: September 17, 1862
kind: battle
relationship: PART_OF
related_event: Maryland Campaign
related_date: September 4–20, 1862
</item 8>

<item 9>
Graph query row:
event: First Battle of Guam
date: 8 December to 10 December 1941
kind: battle
relationship: PART_OF
related_event: Pacific War
related_date: None
</item 9>

<item 10>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 10>
=== END ITEM id=c09389abac28b88b ===

=== ITEM id=a6757f62f77f7ebf ===
Gold sentences and facts:
Sentence 0: Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
  Fact 0.0: Carly Gullickson was born on November 26, 1986.
  Fact 0.1: Carly Gullickson is a former American professional tennis player.
Sentence 1: Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.
  Fact 1.0: Francesca Schiavone was born on 23 June 1980 in Milan.
  Fact 1.1: Francesca Schiavone is an Italian tennis player.
  Fact 1.2: Francesca Schiavone turned professional in 1998.

Retrieved items:
<item 1>
2006 Qatar Ladies Open – Doubles
Alicia Molik and Francesca Schiavone were the defending champions, but Molik did not compete this year.

 Schiavone teamed up with Květa Peschke and lost in querterfinals to Elena Likhovtseva and Vera Zvonareva.
</item 1>

<item 2>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 2>

<item 3>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 3>

<item 4>
2010 US Open – Mixed Doubles
Carly Gullickson and Travis Parrott were the defending champions, but lost in the first round to Gisela Dulko and Pablo Cuevas 2–6, 4–6.
</item 4>

<item 5>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 5>

<item 6>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 6>

<item 7>
Nicole Strausak
Nicole Strausak (born October 21, 1971) is a Swiss curler who retired from competitive curling in March 2005.

 Amongst other awards, she received a World Junior Silver medal in 1991, a European Silver medal in 1993 and a World Silver medal in 2000.

 In 1999, at the European Curling Championships in Chamonix, as well as receiving a Bronze medal, she was awarded the Fair Play & Friendly Award by her fellow
athletes for her conduct and attitude, on and off the ice.
</item 7>

<item 8>
2006 Generali Ladies Linz – Doubles
Gisela Dulko and Květa Peschke were the defending champions, but competed this year with different partners.

 Dulko teamed up with Michaëlla Krajicek and lost in first round to Eleni Daniilidou and Jasmin Wöhr, while Peschke teamed up with Francesca Schiavone and had
to withdrew in their semifinal match (against Corina Morariu and Katarina Srebotnik) as Schiavone had her right knee injured.
</item 8>

<item 9>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 9>

<item 10>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 10>
=== END ITEM id=a6757f62f77f7ebf ===
