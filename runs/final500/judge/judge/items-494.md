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

=== ITEM id=03e65bc7580720a0 ===
Gold sentences and facts:
Sentence 0:  All matches were played at HC Davos's home arena, Eisstadion Davos.
  Fact 0.0: All matches were played at the home arena of HC Davos.
  Fact 0.1: The home arena of HC Davos is Eisstadion Davos.
Sentence 1: Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.
  Fact 1.0: Hockey Club Davos is a professional Swiss ice hockey club.
  Fact 1.1: Hockey Club Davos is based in Davos, Switzerland.

Retrieved items:
<item 1>
2016 Spengler Cup
The 90th Spengler Cup was an ice hockey competition being held in Davos, Switzerland between 26 and 31 December 2016.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 1>

<item 2>
Vaillant Arena
Vaillant Arena (Eisstadion Davos) is an arena in Davos, Switzerland.

 It is primarily used for ice hockey and is the home arena of HC Davos.

 It holds 7,080 people, of which 3,280 are seated.

 Every year the Spengler Cup is played in this arena.
</item 2>

<item 3>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 3>

<item 4>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 4>

<item 5>
Fife Flyers
Fife Flyers Ice Hockey Club is the oldest professional ice hockey club in the UK, established in 1938.

 The Flyers play their home games at Fife Ice Arena in Kirkcaldy which has a capacity of just Over 3000 (seated and standing).

 The arena is home to not only Fife Flyers, who play in the Elite Ice Hockey League but also to Kirkcaldy Junior Ice Hockey Club.

 Traditionally many of the Flyers' players have come up through the junior ranks to play at a professional level.
</item 5>

<item 6>
List of Storhamar Dragons seasons
The Storhamar Dragons are a Norwegian ice hockey club based in Hamar.

 They are members of the highest Norwegian ice hockey league, Eliteserien (known as GET-ligaen for sponsorship reasons).

 Their home arena has been Hamar OL-Amfi since 1992.

 Ice hockey in Hamar was first played in 1955 by boys who had been inspired by the 1952 Winter Olympics in Oslo.

 This became the basis of Storhamar Ishockey, which was founded in 1957 as a department of the sports club Storhamar IL.

 Since 1998, they have been marketing themselves as the Storhamar Dragons.

 As of 2010, they have completed thirty seasons in the Eliteserien, winning 670 regular season games and seven league titles.
</item 6>

<item 7>
Spengler Cup
The Spengler Cup is an annual invitational ice hockey tournament held in Davos, Switzerland.

 First held in 1923, the Spengler Cup is often cited as the oldest invitational ice hockey tournament in the world.

 The event is hosted by the Swiss team HC Davos and played each year in Davos, Switzerland, between Christmas (December 25) and New Year's Day.

 Currently, all games are held at Vaillant Arena.
</item 7>

<item 8>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 8>

<item 9>
2015 Spengler Cup
The 89th Spengler Cup was an ice hockey competition held in Davos, Switzerland between 26 and 31 December 2015.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 9>

<item 10>
HC Davos
Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.

 Davos plays in the National League (NL).

 The team is usually a strong force in the league and often bolster their roster with Swiss national team players and players who once played for National
Hockey League teams.

 HC Davos also has hosted the annual Spengler Cup tournament, an invitational competition, since 1923.
</item 10>
=== END ITEM id=03e65bc7580720a0 ===

=== ITEM id=54402cca68e35bd7 ===
Gold sentences and facts:
Sentence 0:  He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.
  Fact 0.0: He played college football at Hampton University.
  Fact 0.1: He attended Riverdale High School.
  Fact 0.2: Riverdale High School is in Riverdale, Georgia.
Sentence 1: Riverdale is a city in Clayton County, Georgia, United States.
  Fact 1.0: Riverdale is a city.
  Fact 1.1: Riverdale is in Clayton County, Georgia.
  Fact 1.2: Clayton County, Georgia is in the United States.

Retrieved items:
<item 1>
Lake Worth Community High School
Lake Worth Community High School is a B State of Florida rated public high school located in Lake Worth, Florida.

 Established in 1922 as Lake Worth High School, it is currently one of Palm Beach County's largest schools.

 It has six magnet programs: the award-winning Air Force JROTC led by former Multi-National Award-winning Drill Teams, Tyranny and Fidelis; Medicine and Allied
Health; Child Care and Development; Criminal Justice; Culinary and Drafting and Design.

 The Palm Beach County School Board added the word Community to the names of all public high schools, including Lake Worth High School, in the 1980s.
</item 1>

<item 2>
Saint Michael the Archangel Church (Monroe, MI)
In 1918 the parish built the present building of St. Michael School which is now a part of Monroe Catholic Elementary Schools.

 The movement to establish the parish started in 1845 by 14 German immigrants who wanted to worship in their native language.

 The only nearby church was St. Mary of the Immaculate Conception right across the river where St. Michael stands today.

 St. Mary's only worshiped in French and wouldn't hold services in other languages like German and Irish.

 The German and Irish families were forced to move to the second floor of the rectory to have mass in their native languages.

 When the German families were able to acquire land they were eager to leave St. Mary's to start a parish of their own.

 They first purchased an empty lot Humphrey Street but the plan was soon abandoned.

 They then purchased the first mayor of Monroe's estate to build the church and it remains the current site of the church.
</item 2>

<item 3>
Two Rivers High School (Arkansas)
Two Rivers High School is a public high school for students in grades 7 through 12 located in Ola, Arkansas, United States.

 Two Rivers High School is administered by the Two Rivers School District.

 The Two Rivers High School and school district serves 650 sqmi of rural communities including Ola, Plainview, Rover, and Casa.

 Two Rivers High School is one of four high schools within Yell County and the sole high school administered by the Two Rivers School District.
</item 3>

<item 4>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 4>

<item 5>
Saint Michael the Archangel Church (Monroe, MI)
Saint Michael the Archangel Church (St. Michael Church) is located on the west side of the city of Monroe, Michigan along the River Raisin.

 It is home to 1,200 families and it is one of most important religious institutes in Monroe County.

 It was founded in the year 1852.

 The present building was built from 1866-1867.

 It is in the Archdiocese of Detroit.

 Its current priest is Rev. Phillip Ching.

 When the parish was first established, the mayor of Monroe palatial residence was remodeled as a temporary church.

 It was used as a church on the first floor and on the second it was used for the school.

 Later it would only be used as St. Michael School.

 Then in 1866 the cornerstone for the present church was laid.

 The large 187 foot steeple wasn't added until 1883.

 In 1874 the 3-story rectory was built east of the church.

 In 1918 the parish built the present building of St.
</item 5>

<item 6>
North Boone Community Unit School District 200
The current principal is Matt Klett.

 All students who graduate from one of the district elementary schools will consolidate at the fifth grade in one school, called North Boone Upper Elementary
School, whose principal is Mike Greenlee;all students move on to North Boone Middle School, headed by Jeremiah Auble, which educates those in grades seven and
eight.

 The last leg of education this district can provide is to those from grade nine to grade twelve; the facility is called North Boone High School, whose
principal is named Jacob Hubert.

 The current superintendent is Steven Baule.

 The mascot of the district high school is the Viking.
</item 6>

<item 7>
Dannel McCollum
Dannel McCollum was the mayor of Champaign, Illinois from 1987-1999.

 He is the author of the book "The Lord Was Not on Trial" about his mother Vashti McCollum's landmark 1948 Supreme Court case McCollum v. Board of Education,
which struck down religious education in public schools.

 He also wrote the book "Remembering Champaign County".
</item 7>

<item 8>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 8>

<item 9>
Rich Central High School
Rich Central High School or RCHS is a public four-year high school located in Olympia Fields, Illinois, a southern suburb of Chicago, Illinois, in the United
States.

 The Rich Central Campus serves the cities of Matteson, Richton Park, Country Club Hills, Chicago Heights, Tinley Park, and parts of Olympia Fields.

 It is a part of Rich Township District 227, which also includes Rich East High School and Rich South High School.

 Although the school is located in Olympia Fields, it does not serve the entire village.

 Some Olympia Fields students attend high school at Homewood-Flossmoor High School and Bloom High School, depending on the subdivision where the student
resides.
</item 9>

<item 10>
Michael Ola
Michael Ola (born April 19, 1988) is an American football offensive tackle for the Buffalo Bills of the National Football League (NFL).

 He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.

 He has also been a member of the Jacksonville Sharks, Montreal Alouettes, Miami Dolphins, Chicago Bears, San Diego Chargers, Detroit Lions, Seattle Seahawks
and New York Giants.
</item 10>
=== END ITEM id=54402cca68e35bd7 ===

=== ITEM id=04bf9e5f9447e7b0 ===
Gold sentences and facts:
Sentence 0: And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
  Fact 0.0: And There's More is a comedy sketch show.
  Fact 0.1: And There's More stars Jimmy Cricket.
  Fact 0.2: And There's More was produced by Central Independent Television.
  Fact 0.3: And There's More was produced for ITV.
  Fact 0.4: And There's More was produced from 1985 till 1988.
Sentence 1:  He currently lives with his family in Rochdale, North West England.
  Fact 1.0: The person described in this passage currently lives with his family in Rochdale, North West England.
  Fact 1.1: The person described in this passage currently lives in Rochdale.
  Fact 1.2: Rochdale is in North West England.

Retrieved items:
<item 1>
Hotel Kenmore Hall
Hotel Kenmore Hall is a 22-story single room occupancy hotel located at 145 East 23rd Street in the Gramercy section of Manhattan, designed by architect Maurice
Deutsch and constructed in 1927.

 Author Nathanael West lived and worked at the hotel as a night manager in the early years after the hotel opened; one of West's real-life experiences at the
hotel inspired the incident between Romola Martin and Homer Simpson that would later appear in "The Day of the Locust" (1939).

 West allowed friends like Edmund Wilson, Erskine Caldwell, S. J. Perelman and Maxwell Bodenheim free room and meals.

 Dashiell Hammett finished "The Maltese Falcon" here
</item 1>

<item 2>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 2>

<item 3>
Standard Star Building
The Standard Star Building is a historic commercial structure located in the Downtown section of New Rochelle, Westchester County, New York.

 The building, designed by architect Lawrence J. Barnard, was completed in 1924 and is an architecturally significant example of the Italian Renaissance style
in New Rochelle.

 Although it has been altered, these changes occurred only on the Le Count Place façade and the interior of the building.

 The original Standard Star building exterior remains largely unchanged.

 It is further historically significant for its association with a long-published New Rochelle newspaper covering life in New Rochelle.
</item 3>

<item 4>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 4>

<item 5>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 5>

<item 6>
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
</item 6>

<item 7>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 7>

<item 8>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 8>

<item 9>
Live in Old Smokey
Live in Old Smokey is a live CD by English singer/songwriter Linda Lewis recorded at Ronnie Scott's in London and issued in 2006.
</item 9>

<item 10>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 10>
=== END ITEM id=04bf9e5f9447e7b0 ===

=== ITEM id=0025710997fd0a3f ===
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
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 2>

<item 3>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 3>

<item 4>
Red's Dream
Red's Dream is a 1987 American computer-animated short film produced by Pixar and directed by John Lasseter.

 The short film, which runs four minutes, stars Red, a unicycle.

 Propped up in the corner of a bicycle store on a rainy night, Red dreams about a better place.

 "Red's Dream" was Pixar's second computer-animated short following "Luxo Jr." in 1986, also directed by Lasseter.
</item 4>

<item 5>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 5>

<item 6>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 6>

<item 7>
Luxo Jr.
Luxo Jr. is a 1986 American computer-animated short film produced by Pixar and directed by John Lasseter.

 The two-minute short film revolves around one larger and one smaller desk lamp.

 The larger lamp, named Luxo Sr., looks on while the smaller, "younger" Luxo Jr.

 plays exuberantly with a ball that it accidentally deflates.

 "Luxo Jr." was Pixar's first animation after Ed Catmull and John Lasseter left Industrial Light and Magic's computer division.

 It is the source of the hopping desk lamp included in Pixar's corporate logo.
</item 7>

<item 8>
John Lasseter
John Alan Lasseter (born January 12, 1957) is an American animator, film director, screenwriter, and film producer.

 He currently is the chief creative officer of Pixar Animation Studios, Walt Disney Animation Studios, and DisneyToon Studios.

 He is also the Principal Creative Advisor for Walt Disney Imagineering.
</item 8>

<item 9>
Toy Story 3
Toy Story 3 is a 2010 American 3D computer-animated comedy-drama film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the third installment in Pixar's "Toy Story" series, and the sequel to 1999's "Toy Story 2".

 It was directed by Lee Unkrich, the editor of the first two films and the co-director of "Toy Story 2", written by Michael Arndt, while Unkrich wrote the story
along with John Lasseter and Andrew Stanton, respectively director and co-writer of the first two films.

 The plot focuses on the toys Woody, Buzz Lightyear, and their friends dealing with an uncertain future as their owner, Andy, prepares to leave for college.
</item 9>

<item 10>
Lasseter Family Winery
Lasseter Family Winery is a winery located in Glen Ellen, Sonoma County, California.

 The winery was founded in 2000 by Pixar and Walt Disney Animation Studios CCO John Lasseter and his wife, Nancy Lasseter.

 The winery, once inhabited by the Grand Cru Winery, produces approximately 1,200 cases of French red wine blends annually, with the capacity to produce up to
6,000.

 The winery grows Bordeaux and Rhône varietals on 27 acres.

 One of the Lasseters' winemaking mentors was Jess Jackson, of Kendall-Jackson.
</item 10>
=== END ITEM id=0025710997fd0a3f ===

=== ITEM id=2cbacc949180f431 ===
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
Isaac Asimov's Robots and Aliens
Isaac Asimov's Robots and Aliens is a series of novels written by various authors and loosely connected to Isaac Asimov's "Robot" series.

 Each volume is complete in itself, but they form a continuing series.

 The series follows the action of the novels of the "Isaac Asimov's Robot City" series, with the same protagonists Derec and Ariel, and many other characters.

 The plot deals with the Three Laws and encounters between robots and different varieties of alien life.
</item 1>

<item 2>
Isaac Asimov
Isaac Asimov ( ; born Isaak Ozimov; January 2, 1920 – April 6, 1992) was a Russian-American writer and professor of biochemistry at Boston University.

 He was known for his works of science fiction and popular science.

 Asimov was a prolific writer, and wrote or edited more than 500 books and an estimated 90,000 letters and postcards. His books have been published in 9 of the
10 major categories of the Dewey Decimal Classification.
</item 2>

<item 3>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
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
The Last Question
"The Last Question" is a science fiction short story by American writer Isaac Asimov.

 It first appeared in the November 1956 issue of "Science Fiction Quarterly" and was anthologized in the collections "Nine Tomorrows" (1959), "The Best of Isaac
Asimov" (1973), "Robot Dreams" (1986), the retrospective "Opus 100" (1969), and in "Isaac Asimov: The Complete Stories, Vol.

 1" (1990).

 It was Asimov's favorite short story of his own authorship, and is one of a loosely connected series of stories concerning a fictional computer called
Multivac.

 The story overlaps science fiction, theology, and philosophy.
</item 5>

<item 6>
Michael Cunningham
Michael Cunningham (born November 6, 1952) is a U.S. novelist and screenwriter.

 He is best known for his 1998 novel "The Hours", which won the Pulitzer Prize for Fiction and the PEN/Faulkner Award in 1999.

 Cunningham is a senior lecturer of creative writing at Yale University.
</item 6>

<item 7>
Isaac Asimov's Robot City: Perihelion
Isaac Asimov's Robot City: Perihelion is a book written in 1988 by William F. Wu.

 It is part of the series Isaac Asimov's Robot City, which was inspired by Isaac Asimov's "Robot" series.
</item 7>

<item 8>
Isaac Asimov's Robot City: Odyssey
Isaac Asimov's Robot City: Odyssey is a young adult science fiction novel written in 1987 by Michael P. Kube-McDowell.

 It is part of the series Isaac Asimov's Robot City, inspired by Isaac Asimov's "Robot" series.

 The 1995 computer game Robot City is based on the plot of "Odyssey".
</item 8>

<item 9>
Isaac Asimov's Magical Worlds of Fantasy
Isaac Asimov's Magical Worlds of Fantasy is a series of twelve themed paperback fantasy and science fiction anthologies edited by Isaac Asimov, Martin H.
Greenberg and Charles G. Waugh, a companion set to the ten volume "Isaac Asimov's Wonderful Worlds of Science Fiction", produced by the same editors.

 It was published by Signet/New American Library from 1983 to 1991.

 Volumes 1 and 2 were also issued in hardcover in an omnibus collection titled "Isaac Asimov's Magical Worlds of Fantasy: Witches & Wizards".
</item 9>

<item 10>
Graph query row:
person: Isaac Asimov
nationality: American
occupation: writer
nationality_place: None
</item 10>
=== END ITEM id=2cbacc949180f431 ===

=== ITEM id=64131c0a7076ddb7 ===
Gold sentences and facts:
Sentence 0:  In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
  Fact 0.0: In Crash, there is no betting, unlike in Brag.
  Fact 0.1: In Crash, players aim to reach a total of 11 points.
  Fact 0.2: In Crash, the 11 points are gained over successive deals.
Sentence 1: Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.
  Fact 1.0: Three-card brag is a 16th-century British card game.
  Fact 1.1: Three-card brag is the British national representative of the vying or 'bluffing' family of gambling games.

Retrieved items:
<item 1>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 1>

<item 2>
Crash (card game)
Crash is a British card game extension of Nine-card Brag.

 In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
</item 2>

<item 3>
Put (card game)
Put is an English tavern trick-taking card game first recorded in the 16th century and later castigated by 17th century moralists as one of ill repute.

 It belongs to a very ancient family of card games and clearly relates to a group known as Trut, "Truque", also "Tru", and the South American game Truco.

 Its more elaborate version is the Spanish game of Truc, which is still much played in many parts of Southern France and Spain.
</item 3>

<item 4>
Écarté
Écarté is a two-player card game originating from France, the word literally meaning "discarded".

 It is a trick-taking game, similar to whist, but with a special and eponymous discarding phase.

 It is closely related to Euchre, a card game played mainly in the United States.

 Écarté was popular in the 19th century, but is now rarely played.
</item 4>

<item 5>
Triomphe
Triomphe (French for triumph) is a card game dating from the late 15th century.

 It most likely originated in France or Spain (as triunfo) and later spread to the rest of Europe.

 When the game arrived in Italy, it shared a similar name with the pre-existing game and deck known as "trionfi" (tarot).

 While trionfi has a fifth suit that acts as permanent trumps, triomphe randomly selects one of the existing four suits as trumps.

 Another common feature of this game is the robbing of the stock.

 Triomphe became so popular that during the 16th century the earlier game of trionfi was gradually renamed tarocchi, tarot, or tarock.

 This game is the origin of the English word "trump" and is the ancestor of many trick-taking games like Euchre (via Écarté) and Whist (via Ruff and Honours).
</item 5>

<item 6>
Ombre
The historical importance of Ombre in the field of playing cards is the fact that it was the first card game in which a trump suit was established by bidding
rather than by the random process of turning the first card of the stock.

 This game developed from Triunfo, though it was from L'Hombre that the idea of bidding was adopted into other card games such as Skat, and Tarot, which owes
Hombre a good portion of its betting system as well.

 The game continued to be in vogue almost in every corner of Europe from the late 17th through the 18th centuries.
</item 6>

<item 7>
Three card brag
Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.

 Brag is a direct descendant of the Elizabethan game of Primero and one of the several ancestors to poker, just varying in betting style and hand rankings.
</item 7>

<item 8>
Egyptian Ratscrew
Egyptian Ratscrew (also known as Egyptian Ratkiller, Egyptian War, and by other names) is a card game of the matching family of games.

 The game is similar to the 19th century British card game Beggar-My-Neighbour, with the added concept of "slapping" cards when certain combinations are played,
similar to and perhaps borrowed from Slapjack.
</item 8>

<item 9>
Gleek (card game)
Gleek is an English card game for three persons.

 It is played with a 44-card pack and was popular from the 16th century through the 18th century.
</item 9>

<item 10>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 10>
=== END ITEM id=64131c0a7076ddb7 ===

=== ITEM id=c1aa6326a869a384 ===
Gold sentences and facts:
Sentence 0: Chester Erskine (November 29, 1905 – April 7, 1986) was a Hollywood and Broadway director, writer, and producer.
  Fact 0.0: Chester Erskine was born on November 29, 1905.
  Fact 0.1: Chester Erskine died on April 7, 1986.
  Fact 0.2: Chester Erskine was a Hollywood and Broadway director.
  Fact 0.3: Chester Erskine was a Hollywood and Broadway writer.
  Fact 0.4: Chester Erskine was a Hollywood and Broadway producer.
Sentence 1: Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.
  Fact 1.0: Sam Taylor was born on August 13, 1895, in New York City.
  Fact 1.1: Sam Taylor died on March 6, 1958, in Santa Monica.
  Fact 1.2: Sam Taylor was a film director.
  Fact 1.3: Sam Taylor was a screenwriter.
  Fact 1.4: Sam Taylor was a film producer.
  Fact 1.5: Sam Taylor was most active in the silent film era.

Retrieved items:
<item 1>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 1>

<item 2>
Buford Abner
James Buford Abner (November 10, 1917 - November 19, 2011) was an American songwriter, musician and singer who worked during the early days of country music,
working in both secular and gospel country music genres.

 With his brother Merle Abner, his uncle Stacy Abner, George Hughes and Billy Carrier, he was a member of the Swanee River Boys.

 He was inducted into the Southern Gospel Music Association Hall of Fame in 2002 and the Atlanta Country Music Hall of Fame as a member of the Swanee River
Boys.
</item 2>

<item 3>
Chester Erskine
Chester Erskine (November 29, 1905 – April 7, 1986) was a Hollywood and Broadway director, writer, and producer.
</item 3>

<item 4>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 4>

<item 5>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 5>

<item 6>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 6>

<item 7>
Mary Elizabeth Bliss
Mary Elizabeth "Betty" Taylor Bliss Dandridge, born Mary Elizabeth Taylor (April 20, 1824 – July 25, 1909), was the youngest of the three surviving daughters of
President Zachary Taylor (1849-1850) and Margaret Mackall Smith Taylor.
</item 7>

<item 8>
Kate McTell
Kate McTell (born Ruthy Kate Williams; August 22, 1911 – October 3, 1991) was an American blues musician and nurse from Jefferson County, Georgia.

 She is known primarily as the former wife of the blues musician Blind Willie McTell, whom she accompanied vocally on several recordings.

 She may have recorded as Ruby Glaze, but there is some uncertainty about whether she and Glaze were the same person, despite the fact that she claimed to be
Glaze.
</item 8>

<item 9>
Speedy West
Wesley Webb West (January 25, 1924 – November 15, 2003), better known as Speedy West, was an American pedal steel guitarist and record producer.

 He frequently played with Jimmy Bryant, both in their own duo and as part of the regular Capitol Records backing band for Tennessee Ernie Ford and many others.

 He also played on Loretta Lynn's first single.
</item 9>

<item 10>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 10>
=== END ITEM id=c1aa6326a869a384 ===

=== ITEM id=be04e9f6f8a1f539 ===
Gold sentences and facts:
Sentence 0: Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on science and technology subjects.
  Fact 0.0: Popular Science is also known as PopSci.
  Fact 0.1: Popular Science is an American bi-monthly magazine.
  Fact 0.2: Popular Science carries popular science content.
  Fact 0.3: Popular science content refers to articles for the general reader on science and technology subjects.
Sentence 1: Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.
  Fact 1.0: Girlfriends was a women's magazine.
  Fact 1.1: Girlfriends provided critical coverage of culture.
  Fact 1.2: Girlfriends provided critical coverage of entertainment.
  Fact 1.3: Girlfriends provided critical coverage of world events.
  Fact 1.4: Girlfriends' coverage was from a lesbian perspective.
Sentence 2:  Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.
  Fact 2.0: Girlfriends was published monthly from San Francisco since 1993.
  Fact 2.1: Girlfriends was distributed nationwide by Disticor.
Sentence 3:  "Girlfriends" magazine ceased publication in 2006.
  Fact 3.0: Girlfriends magazine ceased publication in 2006.

Retrieved items:
<item 1>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 1>

<item 2>
Popular Science
Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on
science and technology subjects.

 "Popular Science" has won over 58 awards, including the American Society of Magazine Editors awards for its journalistic excellence in both 2003 (for General
Excellence) and 2004 (for Best Magazine Section).

 With roots beginning in 1872, "Popular Science" has been translated into over 30 languages and is distributed to at least 45 countries.
</item 2>

<item 3>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 3>

<item 4>
Computer Magazine
Computer Magazine (and their website www.ComputerMagazine.com) is a popular magazine and online news site on computing and technology, offering current news and
reviews of popular and new business and consumer technologies, software, hardware, mobile computing, tablets, PCs, Macs, Windows, Linux, telecom, cellular,
wireless, data, cloud and science news on digital technologies and everything in the "tech-sphere and digi-verse", especially focused on information technology,
devices, software and services and related subjects, such as networking, servers, data centers and corporate data infrastructure technologies, and the Internet.

 Their online site, since 1997, is located at ComputerMagazine.com.

 "Computer Magazine" produces industry instructional and a popular ongoing webcast/podcast talk show and performs evaluations and reviews of IT industry
technology products, hardware, software and services with objective reporting widely respected as independent and objective, and trusted in the industry.
</item 4>

<item 5>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 5>

<item 6>
Popular science
Popular science (also pop-science or popsci) is interpretation of science intended for a general audience.

 While science journalism focuses on recent scientific developments, popular science is more broad-ranging.

 It may be written by professional science journalists or by scientists themselves.

 It is presented in many forms, including books, film and television documentaries, magazine articles, and web pages.
</item 6>

<item 7>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 7>

<item 8>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 8>

<item 9>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 9>

<item 10>
Computer Magazine
"Computer Magazine" is a free publication (in addition to their webcasts and other resources) sponsored by the nonprofit UTCP (United Technology and Computing
Professionals) organization, and as such charges no fees for the publication nor is influenced by advertising, so their reviews are relied on in the industry
and considered unbiased and thorough.

 "Computer Magazine" is one of the early large technology publications and resources available on the web still existent and thriving today and that has
remained independent.

 ComputerMagazine.com is a tech news and resources consolidator that publishes part of the site in a semi-time line/blogging format that is popular among their
wide following of subscriber and non-subscriber readers, allowing readers to respond and comment on various articles.

 Site contributors include many of the well known technology authors, experts and publication sources, content and articles are provided by major technology
syndicators and by external expert technology sources (such as "Computer World", "Information Week", "Network World", "Wired," "Time", etc.)
</item 10>
=== END ITEM id=be04e9f6f8a1f539 ===

=== ITEM id=931ace3bff0a982f ===
Gold sentences and facts:
Sentence 0: Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.
  Fact 0.0: Mirrors is a 2008 supernatural horror film.
  Fact 0.1: Mirrors was directed by Alexandre Aja.
  Fact 0.2: Kiefer Sutherland starred in Mirrors.
  Fact 0.3: Paula Patton starred in Mirrors.
  Fact 0.4: Amy Smart starred in Mirrors.
Sentence 1: Paula Maxine Patton (born December 5, 1975) is an American actress.
  Fact 1.0: Paula Maxine Patton was born on December 5, 1975.
  Fact 1.1: Paula Maxine Patton is an American actress.

Retrieved items:
<item 1>
Ela, Hela and The Hitch
Ela, Hela, and the Hitch is a play by Václav Havel.

 The play was written for the Artistic Director of the Theatre on the Balustrade, Ivan Vyskočil, as part of a longer evening, entitled "Hitchhiking".

 Along with "Ela, Hela, and the Hitch", Havel also wrote a sketch called "Motormorphosis".

 Reportedly, Vyskočil altered Havel’s sketches for the performance, though the original text was discovered by a Czech theater scholar, Lenka Jungmannová.

 "Motormorphosis", in a translation by Carol Rocamora, was performed at the Havel Festival in 2006, a world premiere of the text as written.

 "Ela, Hela, and the Hitch" premiered in an English translation by Edward Einhorn following a revival of "Motormorphosis" at New York’s Bohemian National Hall
in 2011.
</item 1>

<item 2>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 2>

<item 3>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 3>

<item 4>
Paula Patton
Paula Maxine Patton (born December 5, 1975) is an American actress.

 Patton made her film debut in the 2005 comedy film "Hitch" and starred in the epic fantasy film "Warcraft" (2016), based on the game series of the same name.

 Patton has also been the female lead in "Déjà Vu" (2006), "" (2011) and "2 Guns" (2013), and appeared in the critically acclaimed "Precious" (2009).
</item 4>

<item 5>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 5>

<item 6>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 6>

<item 7>
Hitch (film)
Hitch is a 2005 American romantic comedy film directed by Andy Tennant and starring Will Smith.

 The film, which was written by Kevin Bisch, co-stars Eva Mendes, Kevin James, and Amber Valletta.

 Smith plays the main fictional character of the film, Alex "Hitch" Hitchens, who is a professional dating consultant who makes a living teaching men how to woo
women.

 The film was released on February 11, 2005 by Columbia Pictures.
</item 7>

<item 8>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 8>

<item 9>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 9>

<item 10>
Mirrors (film)
Mirrors is a 2008 supernatural horror film directed by Alexandre Aja, starring Kiefer Sutherland, Paula Patton, and Amy Smart.

 The film was first titled "Into the Mirror", but the name was later changed to "Mirrors".

 Filming began on May 1, 2007, and it was released in American theaters on August 15, 2008.
</item 10>
=== END ITEM id=931ace3bff0a982f ===

=== ITEM id=7cbdc7750284c70b ===
Gold sentences and facts:
Sentence 0:  Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
  Fact 0.0: Cacareco's candidacy inspired the Rhinoceros Party of Canada.
  Fact 0.1: The Rhinoceros Party of Canada was nominally led by the rhinoceros Cornelius the First.
Sentence 1: The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.
  Fact 1.0: The Rhinoceros Party's French name is 'Parti Rhinoceros'.
  Fact 1.1: The Rhinoceros Party was a registered political party in Canada.
  Fact 1.2: The Rhinoceros Party was a registered political party in Canada from the 1960s to the 1990s.

Retrieved items:
<item 1>
Politics of Canada
The politics of Canada function within a framework of parliamentary democracy and a federal system of parliamentary government with strong democratic
traditions.

 Canada is a constitutional monarchy, in which the Monarch is head of state.

 The country has a multi-party system in which many of its legislative practices derive from the unwritten conventions of and precedents set by the United
Kingdom's Westminster Parliament.

 However, Canada has evolved variations: party discipline in Canada is stronger than in the United Kingdom and more parliamentary votes are considered motions
of confidence, which tends to diminish the role of non-Cabinet Members of Parliament, (MPs).

 Such members, in the government caucus, and junior or lower-profile members of opposition caucuses, are known as backbenchers. Backbenchers can, however, exert
their influence by sitting in parliamentary committees, like the Public Accounts Committee or the National-Defence Committee.
</item 1>

<item 2>
Rhinoceros Party of Canada (1963–93)
The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.

 Operating within the tradition of political satire, the Rhinoceros Party's basic credo, their so-called primal promise, was "a promise to keep none of our
promises".

 They then promised outlandishly impossible schemes designed to amuse and entertain the voting public.
</item 2>

<item 3>
Cacareco
Cacareco (1954-1962) was a female black rhinoceros exhibited in Brazilian zoos.

 She became famous as a candidate for the 1958 São Paulo city council elections with the intention of protesting against political corruption.

 Electoral officials did not accept Cacareco's candidacy, but she eventually won 100,000 votes, more than any other party in that same election (which was also
marked by rampant absenteeism).

 Today, the term "voto Cacareco" (Cacareco vote) is commonly used to describe protest votes in Brazil.

 Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
</item 3>

<item 4>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 4>

<item 5>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 5>

<item 6>
National Advancement Party of Canada
The National Advancement Party of Canada (French: "Parti de l'Essor National du Canada" ) is a minor federal political party in Canada.

 The party describes itself as for "advancement of the Canadian people and Canada".

 The party was formed in 2014 and registered with Elections Canada in 2015, initially under the name Democratic Advancement Party of Canada.

 The party is headquartered in Calgary, Alberta and led by writer Stephen Garvey.
</item 6>

<item 7>
The New Canada
The New Canada is a Canadian political literature book written by Reform Party of Canada founder and leader Preston Manning and published by Macmillan Canada.

 The book explains the personal, religious, and political life of Preston Manning and explains the roots and beliefs of the Reform Party.

 At the time of its publishing in 1992, Reform had become a popular populist conservative party in Western Canada after the mainstream Progressive Conservative
Party of Canada was collapsing in support and in 1991 decided to expand eastward into Ontario and the Maritimes.

 One year later the PC party collapsed in the 1993 federal election, allowing the Reform Party to make political history in Canada, deplacing the PCs as the
dominant conservative party in Canada.

 Reform, later renamed the Canadian Alliance, merged with the PC Party in 2003, to form a united right-wing alternative to the governing Liberal Party of
Canada, named the Conservative Party of Canada which has dropped many of the populist themes that the Reform Party had.
</item 7>

<item 8>
Aboriginal Peoples Party of Canada
The Aboriginal Peoples Party of Canada (APP) is a Canadian political party that was founded in 2005.

 The party was conceived by University of Lethbridge student Myron Wolf Child.

 It held its founding meeting on August 21, 2005, in St. Albert, Alberta.

 The APP was headed by interim leader Bill Montour, a former Chief of the Six Nations Reserve in Ontario.

 In October 2005, the party had 122 confirmed members, just less than half the number needed to register as an official party in Canada.

 In late October 2005, the APP sought to unite with the First Peoples National Party of Canada which also had fewer than the number of confirmed members needed
to become a registered political party.

 The First Peoples National Party of Canada became an eligible political party on December 6, 2005.

 Whether or not this was accomplished through a merger with the APP is unclear.

 If this is the case, the APP no longer exists, but, as one of the stipulations of a possible merger was that a name for the new party would be determined at
the party's first convention, it is possible that the name Aboriginal Peoples Party of Canada will come into usage again.
</item 8>

<item 9>
Western Canada Miners
The Western Canada Miners (also known as just the Canada Miners) are a professional independent baseball team based in Yuma, Arizona, and representing western
Canada.

 They play in the developmental Arizona Winter League, a short-season instructional winter league affiliated with the North American League and they play their
home games at Desert Sun Stadium in Yuma, as well as the San Diego Surf Dawgs, Saskatchewan Silver Sox, Team Canada and Yuma Scorpions.

 The team uses the logo and uniforms of the now-defunct Mesa Miners team and are owned by Diamond Sports & Entertainment.
</item 9>

<item 10>
Revolutionary Communist Party of Canada
The Revolutionary Communist Party of Canada, or RCP Canada, is a revolutionary communist party oriented around Marxism-Leninism-Maoism.

 The creation of the organizational stage of the Party was adopted at what was called the "Revolutionary Communist Conference", which was held in Montreal,
Quebec in November 2000 by activists and former members of the labour union movements and youth organizations of the 1960s, 1970s and 1980s, who felt that the
revolutionary situation in Canada warranted the creation of a Party dedicated to a communist revolution.

 The party was initially called the Revolutionary Communist Party (Organizing Committees) (RCP(OC)).

 At this conference, participants adopted the Party's first Draft Programme.

 Because of the location of the conference, the majority of the founding members were French-speaking Quebecers, and the Party began an extensive effort to
reach out to the rest of Canada, starting with the Canadian Revolutionary Congress held in November 2006 in Toronto.

 It is not recognized as a political party by Elections Canada, because the party rejects what it calls the "bourgeois electoral system" and doesn't seek
recognition by the State.
</item 10>
=== END ITEM id=7cbdc7750284c70b ===

=== ITEM id=a5e630e82a1cdb8c ===
Gold sentences and facts:
Sentence 0:  Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
  Fact 0.0: Akrofuom is located in the Obuasi Municipal district.
  Fact 0.1: The Obuasi Municipal district is in the Ashanti Region of Ghana.
Sentence 1: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.
  Fact 1.0: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

Retrieved items:
<item 1>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 1>

<item 2>
Western Region (Ghana)
The Western Region is located in south Ghana, spreads from the Ivory Coast border in the west to the Central region in the east, includes the capital and large
twin city of Sekondi-Takoradi on the coast, coastal Axim, and a hilly inland area including Elubo.

 It includes Ghana's southernmost location, Cape Three Points, where crude oil was discovered in commercial quantities in June 2007.

 The Western Region enjoys a long coastline that stretches from South Ghana's border with Ivory Coast to the Western region's boundary with the Central Region
on the east.
</item 2>

<item 3>
Dodi Papase
Dodi Papase is a town in the Volta Region of Ghana.

 With a population of approximately 5,254 ,Dodi Papase is the second-most-populous city in the Volta region of Ghana.

 It is located in eastern Ghana on the banks of the Asukawkaw River and is known for the Dodi Papase Community Secondary School, a second cycle institution
(senior high school).
</item 3>

<item 4>
Techimantia
Techimantia is a city located in the Tano District of the Brong Ahafo Region of Ghana.

 It is one of the large cities in the region.

 It is believed to be originally inhabited by Hunters for the Ashanti King from Kumasi.

 These Hunters originally settled in the region after they found that there were lots of elephants in the region which could be hunted for the Ashanti King.The
name of the King is Oti Ampem.Currently,the chief of the town is no more(dead).

The occupation of the people in the town is large scale tomatoes farming.Techimantia is one of the largest towns in the region but there are few productive or
government work in the town.This was attributed to the curse from Okomfo Anokye of the Ashanti Kingdom.
</item 4>

<item 5>
Taifa, Accra
Taifa is a town in the Ga East Municipal District, a district in the Greater Accra Region of south-eastern Ghana near the capital Accra.

 Taifa is the twenty-sixth largest settlement in Ghana, in terms of population, with a population of 68,459 people.

 Taifa is located in the northwest suburbs area of Accra.

 It has a breakpoint on a railway line and a small park located on the northern edge of the location of the Taifa Ghana Atomic Energy Commission.

 At the Ghana 2000 census of 26 March 2000, the population was 26,145 inhabitants living in the city.

 Projections of 1 January 2007 estimated the population to be 48,927 inhabitants.

 In the census of 1984 there was only 1,009 inhabitants.

 The strong population growth of the Town is influenced by, among other things, a large number of illegal immigrants from west African countries who move to
towns and villages near the industrial town of Tema, just to find a job.
</item 5>

<item 6>
Tafo
Tafo is a town in Kumasi Metropolitan District in the Ashanti Region of Ghana near the regional capital Kumasi.

 Tafo is the thirtieth most populous settlement in Ghana, in terms of population, with a population of 60,919 people.

 Because of the town's population and housing development in recent years, it is debatable whether Tafo is still regarded as a separate town, or already a
suburb of Kumasi, the capital of the Ashanti region.

 The town is near Kumasi, with a distance of approximately 3.3 kilometers to the center of a similar name sounding village named New Tafo and must be
distinguished from Tafo.

 Tarkwa is located just 4.6 km away from Tafo.

 The city center of Kumasi is located approximately 9.8 kilometers away.

 Tafo is one of the urban constituencies of the Kumasi Metropolitan Assembly, the town's parliamentary candidate shall have one direct seat to the Parliament of
Ghana.
</item 6>

<item 7>
Akrofuom (Ghana parliament constituency)
Akrofuom is one of the constituencies represented in the Parliament of Ghana.

 It elects one Member of Parliament (MP) by the first past the post system of election.

 Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
</item 7>

<item 8>
Tema
Tema is a city on the Bight of Benin and Atlantic coast of Ghana.

 It is located 25 km east of the capital city; Accra, in the region of Greater Accra, and is the capital of the Tema Metropolitan District.

 As of 2013, Tema is the eleventh most populous settlement in Ghana, with a population of approximately 161,612 people – a marked decrease from its 2005 figure
of 209,000.

 The Greenwich Meridian (00 Longitude) passes directly through the city.

 Tema is locally nicknamed the "Harbour Town" because of its status as Ghana's largest seaport.
</item 8>

<item 9>
Obuasi Municipal District
The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

 The capital of the Obuasi Municipality is Obuasi.
</item 9>

<item 10>
Ashanti Region
The Ashanti Region is located in south Ghana and is third largest of 10 administrative regions, occupying a total land surface of 24389 km2 or 10.2 per cent of
the total land area of Ghana.

 In terms of population, however, it is the most populated region with a population of 4,780,380 according to the 2010 census, accounting for 19.4% of Ghana’s
total population.

 The Ashanti Region is known for its major gold bar and cocoa production.

 The largest city and regional capital is Kumasi.
</item 10>
=== END ITEM id=a5e630e82a1cdb8c ===
