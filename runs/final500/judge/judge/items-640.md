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

=== ITEM id=a065fa6534904a75 ===
Gold sentences and facts:
Sentence 0: Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in England to Australian parents.
  Fact 0.0: Ralph Foster Smart was born on 27 August 1908.
  Fact 0.1: Ralph Foster Smart died on 12 February 2001.
  Fact 0.2: Ralph Foster Smart was a film producer.
  Fact 0.3: Ralph Foster Smart was a television producer.
  Fact 0.4: Ralph Foster Smart was a director.
  Fact 0.5: Ralph Foster Smart was a writer.
  Fact 0.6: Ralph Foster Smart was born in England.
  Fact 0.7: Ralph Foster Smart's parents were Australian.
Sentence 1: Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.
  Fact 1.0: Randolph Severn 'Trey' Parker III was born on October 19, 1969.
  Fact 1.1: Randolph Severn 'Trey' Parker III is an American actor.
  Fact 1.2: Randolph Severn 'Trey' Parker III is an animator.
  Fact 1.3: Randolph Severn 'Trey' Parker III is a writer.
  Fact 1.4: Randolph Severn 'Trey' Parker III is a director.
  Fact 1.5: Randolph Severn 'Trey' Parker III is a producer.
  Fact 1.6: Randolph Severn 'Trey' Parker III is a singer.
  Fact 1.7: Randolph Severn 'Trey' Parker III is a songwriter.

Retrieved items:
<item 1>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 1>

<item 2>
Ralph Smart
Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in
England to Australian parents.
</item 2>

<item 3>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 3>

<item 4>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 4>

<item 5>
Dave Kneebone
Dave Kneebone is an American producer.

 Along with Eric Wareheim and Tim Heidecker he runs Abso Lutely Productions which produces a variety of television and films.

 His role has been described as the business chief and "straight man" at Abso Lutely.

 Kneebone has worked as a producer on a variety of television shows including "Comedy Bang!

 Bang!

", "Nathan for You", and "Tim and Eric Awesome Show, Great Job!

".

 He is also listed a producer on the feature film "Tim and Eric's Billion Dollar Movie".
</item 5>

<item 6>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 6>

<item 7>
Trey Parker
Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.

 He is known for co-creating "South Park" (1997–present) along with his creative partner Matt Stone, as well as co-writing and co-directing the Tony
Award-winning musical "The Book of Mormon" (2011).

 Parker was interested in film and music as a child, and attended the University of Colorado, Boulder following high school, where he met Stone.

 The two collaborated on various short films, and starred in a feature-length musical, titled "Cannibal!

 The Musical" (1993).
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
=== END ITEM id=a065fa6534904a75 ===

=== ITEM id=af96ad9c2bdee6e2 ===
Gold sentences and facts:
Sentence 0: The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of its kind in Northern Europe.
  Fact 0.0: The Cathedral and Metropolitical Church of Saint Peter in York is commonly known as York Minster.
  Fact 0.1: York Minster is the cathedral of York, England.
  Fact 0.2: York Minster is one of the largest cathedrals of its kind in Northern Europe.
Sentence 1: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.
  Fact 1.0: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

Retrieved items:
<item 1>
The entries cover British and Australian church heritage. London’s St Paul’s, founded on the site of a church dating to 604, is Wren’s Grade I-listed Baroque
cathedral, rebuilt after the Great Fire. Rockhampton’s St Paul’s Cathedral (1883–1953) and adjacent hall (built circa 1900) are Queensland-listed. Augustine
became Canterbury’s first archbishop in 597. Other cathedrals include medieval Lichfield (England’s only
</item 1>

<item 2>
Minster (church)
Minster is an honorific title given to particular churches in England, most famously York Minster in York, Westminster in London and Southwell Minster in
Southwell.

 The term "minster" is first found in royal foundation charters of the 7th century.

 Although it corresponds to the Latin "monasterium" or monastery, it then designated any settlement of clergy living a communal life and endowed by charter with
the obligation of maintaining the daily office of prayer.

 Widespread in 10th-century Anglo-Saxon England, minsters declined in importance with the systematic introduction of parishes and parish churches from the 11th
century onwards.

 It continued as a title of dignity in later medieval England, for instances where a cathedral, monastery, collegiate church or parish church had originated
with an Anglo-Saxon foundation.

 Eventually a minster came to refer more generally to "any large or important church, especially a collegiate or cathedral church".

 In the 21st century, the Church of England has designated additional minsters by bestowing the status on existing parish churches.
</item 2>

<item 3>
York Minster astronomical clock
The York Minster astronomical clock was installed in the North Transept of York Minster in 1955.

 It was first conceived in 1944 and designed by R d'E Atkinson, chief assistant at the Royal Greenwich Observatory.

 The clock is a memorial to the airmen operating from bases in Yorkshire, Durham, and Northumberland who were killed in action during World War II.
</item 3>

<item 4>
York Minster Police
York Minster Police is a small, specialised cathedral constabulary responsible for policing York Minster in York, United Kingdom.
</item 4>

<item 5>
The entries span UK transport, places, churches and heritage: Rosyth station serves the naval dockyard; Wood Green and Hounsfield Heights/Briar Hill are
described, as are West Midlands railways and Black Country attractions. Church histories include St Paul’s London and Rockhampton, Guildford Cathedral, York
Minster, and churches in Karlsøy, Manhattan, Istanbul, Ballston Spa and Iowa City. Other subjects include marine-freight firms, City Cruises, London landmarks,
ecc
</item 5>

<item 6>
York Minster
The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of
its kind in Northern Europe.

 The minster is the seat of the Archbishop of York, the second-highest office of the Church of England, and is the mother church for the Diocese of York and the
Province of York.

 It is run by a dean and chapter, under the Dean of York.

 The title "minster" is attributed to churches established in the Anglo-Saxon period as missionary teaching churches, and serves now as an honorific title.

 Services in the minster are sometimes regarded as on the High Church or Anglo-Catholic end of the Anglican continuum.
</item 6>

<item 7>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 7>

<item 8>
Old Palace (York)
The Old Palace in the city of York, North Yorkshire, England, is also known as the Minster Library and is in Dean's Park.

 It houses York Minster’s library and archives as well as the Collections Department and conservation studio.

 Its name is a new one and renders homage to the part of the building that used to be the chapel of the Archbishop of York, which was built in the 13th century.

 It was refurbished in 1810 and shortly thereafter became the home of the Minster library.

 Notable items held in the collection include cathedral records dating to back to 1150 and a copy of the 1631 Wicked Bible.

 It is a Grade I listed building.

 An extension was added in 1998.
</item 8>

<item 9>
York Minster, the vast cathedral and seat of the Archbishop of York, is a major historic landmark in York, England. Its 1955 astronomical clock commemorates
Second World War airmen from northern England. A small specialised police force protects the Minster. Nearby, the Grade I listed Old Palace houses the Minster’s
library, archives and conservation facilities; Grays Court, partly dating to 1080, was built for the Minster’s treasurers. “Minster” began
</item 9>

<item 10>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 10>
=== END ITEM id=af96ad9c2bdee6e2 ===

=== ITEM id=f94cb5894ce7cffc ===
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
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 1>

<item 2>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 2>

<item 3>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 3>

<item 4>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 4>

<item 5>
Graph query row:
person1: Dan Hurley
person2: Bobby Hurley
shared_parent: Robert Emmet Hurley, Sr.
head_coach: Bobby Hurley
role: head coach
organization: Arizona State men's team
</item 5>

<item 6>
Rhode Island Rams men's basketball
The Rhode Island Rams men's basketball team is a college basketball program that competes in NCAA Division I and the Atlantic 10 Conference representing the
University of Rhode Island.

 The team is under the direction of head coach Dan Hurley, who was hired on March 20, 2012.

 The Rams play their home games at the Ryan Center (capacity 7,657).
</item 6>

<item 7>
Nate Oats
Nathanael "Nate" Oats (born October 13, 1974) is an American basketball coach.

 He is currently the head basketball coach at the State University of New York at Buffalo, following two seasons as an assistant coach under Bobby Hurley.

 Oats was named head coach on April 11, 2015 after Hurley was hired by Arizona State as head coach.

 In only his first season as head coach, he took the Buffalo Bulls to their second straight NCAA Tournament bid after winning the Mid-American Conference
Championship.
</item 7>

<item 8>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 8>

<item 9>
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
</item 9>

<item 10>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 10>
=== END ITEM id=f94cb5894ce7cffc ===

=== ITEM id=66867707b282603c ===
Gold sentences and facts:
Sentence 0:  He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth Games in Edinburgh, Scotland.
  Fact 0.0: He also participated in the 10,000 metres at the 1967 Pan American Games.
  Fact 0.1: The 1967 Pan American Games were held in Winnipeg, Manitoba, Canada.
  Fact 0.2: He represented Guyana at the 1970 British Commonwealth Games.
  Fact 0.3: The 1970 British Commonwealth Games were held in Edinburgh, Scotland.
Sentence 1:  The race consists of 25 laps around an Olympic-sized track.
  Fact 1.0: The race consists of 25 laps.
  Fact 1.1: The 25 laps are around an Olympic-sized track.

Retrieved items:
<item 1>
Harry Prowell
Harry Prowell A.A.(10 July 1936 – 27 June 2000) was a Guyanese long distance runner who represented Guyana in the Marathon at the 1968 Summer Olympics in Mexico
City, Mexico.

 He is known to be one of the greatest Marathon runners Guyana has ever produced, setting the national record in 1968.

 To date, he is the only Guyanese ever to compete in the Marathon at the Summer Olympic Games and one of the most prominent Indo-Caribbean long distance runners
of his time.

 He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth
Games in Edinburgh, Scotland.
</item 1>

<item 2>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 2>

<item 3>
Dave Bailey (athlete)
He was a member of 9 Canadian National Track and Field Teams competing at World University Games, Budapest 1965 and Tokyo 1967, Commonwealth Games, Kingston,
Jamaica, 1966, Pan American Games, Winnipeg, 1967, Commonwealth vs USA, Los Angeles, 1967, Pre-Olympic Games, Mexico City, 1967, Soviet Union Tour, 1968,
European Tour, 1968, Olympic Games, Mexico City, 1968.
</item 3>

<item 4>
Nora Rocha
Nora Leticia Rocha de la Cruz (born December 18, 1967 in Monclova, Coahuila) is a retired female track and field athlete from Mexico, who competed in the 5000
and 10,000 metres.

 She claimed the gold medal in the women's 10,000 metres at the 1999 Pan American Games in Winnipeg, Canada.
</item 4>

<item 5>
John Lindsay (Paralympian)
At the 2004 Athens Games, he came seventh in the first round of the Men's 100 m T53 event and sixth in the third round of the Men's 200 m T53 event.

 He was an Australian Institute of Sport scholarship holder in 1995 and 2000.
</item 5>

<item 6>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 6>

<item 7>
10,000 metres
The 10,000 metres or 10,000-meter run is a common long-distance track running event.

 The event is part of the athletics programme at the Olympic Games and the World Championships in Athletics and is common at championship level events.

 The race consists of 25 laps around an Olympic-sized track.

 It is less commonly held at track and field meetings, due to its duration.

 The 10,000 metres track race is usually distinguished from its road running counterpart, the 10K run, by its reference to the distance in metres rather than
kilometres.
</item 7>

<item 8>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 8>

<item 9>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 9>

<item 10>
Dave Bailey (athlete)
David Bailey (born March 17, 1945 in Toronto, Ontario) is a retired track and field athlete, who represented Canada at the 1968 Summer Olympics in the men's
1.500 metres.

 He was the first Canadian to run the mile in less than 4 minutes (3:59.1) in San Diego, CA on June 11, 1966 and the first Canadian to the run the mile in less
than 4 minutes in Canada (3:57.7)in Toronto on July 22, 1967.

 A resident of Willowdale, Ontario he won the bronze medal in this event at the 1967 Pan American Games and the silver medal in this event at the 1968 World
University Games He was two-time Canadian Universities Track and Field Athlete of Year (1965, 1967) and two-time inductee into the University of Toronto Sports
Hall of Fame (individually 1998 and team 2003).
</item 10>
=== END ITEM id=66867707b282603c ===

=== ITEM id=3b1811de5b409bdd ===
Gold sentences and facts:
Sentence 0: Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic series.
  Fact 0.0: Levni Yilmaz was born in 1973.
  Fact 0.1: Levni Yilmaz is a San Francisco based independent film maker.
  Fact 0.2: Levni Yilmaz is an artist.
  Fact 0.3: Levni Yilmaz is a publisher.
  Fact 0.4: Levni Yilmaz is best known for his 'Tales of Mere Existence' animated comic series.
Sentence 1: Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.
  Fact 1.0: Pamela B. Green is American.
  Fact 1.1: Pamela B. Green is a film producer.
  Fact 1.2: Pamela B. Green is a film director.
  Fact 1.3: Pamela B. Green is known for her work in feature film titles.
  Fact 1.4: Pamela B. Green is known for her work in motion graphics.

Retrieved items:
<item 1>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 1>

<item 2>
Roger Lancelyn Green
Roger (Gilbert) Lancelyn Green (2 November 1918 – 8 October 1987) was a British biographer and children's writer.

 He was an Oxford academic who formed part of the Inklings literary discussion group along with C. S. Lewis and J. R. R. Tolkien.
</item 2>

<item 3>
Lev Yilmaz
Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic
series.
</item 3>

<item 4>
Pamela B. Green
Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.

 She is the director and producer of the documentary "Be Natural: The Untold Story of Alice Guy-Blache".
</item 4>

<item 5>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 5>

<item 6>
Aryanto Yuniawan
Aryanto Yuniawan (born June 18, 1977 in Magelang; age 40) is Chief Executive Officer of PT Mataram Surya Visi Sinema, a multimedia and animation movie company
located in Jogja Indonesia.

 Ary (his nickname) is best known as a director of Battle of Surabaya, the first 2D animation movie in Indonesia which got some awards i.e. Gold Remi Award USA
2016, GrandPrize Winner of SICAF 2016 in South Korea, Winner of Nice International Filmmaker Festival 2017 in France, Winner of Nioda International Film
Festival in India and others.

 He is also acclaimed as a writer, a producer, and currently working on several articles and books related to his job and hobby.

 He started his career from directing and producing independent animation films including music videos.

 He was trusted to be one of the juries of several film festivals such as Festival Film Indonesia (FFI 2016), Noida International Film Festival (NIFF 2016), and
Seoul International Cartoon and Animation Festival (SICAF 2017).
</item 6>

<item 7>
İsmet Ergün
İsmet Ergün (born 1950) is a Berlin artist and stage designer of Turkish provenance.

 Alongside her theatre work she has worked in the world of cinema as an art director and as a production designer.

 The short film which she directed, and for which she herself wrote the screenplay, was singled out for commendation at the 2007 Locarno Film Festival: it also
won the "Best German Film 2007" at the later the same year.
</item 7>

<item 8>
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
</item 8>

<item 9>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 9>

<item 10>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 10>
=== END ITEM id=3b1811de5b409bdd ===

=== ITEM id=6df2733499d9629a ===
Gold sentences and facts:
Sentence 0: The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in 1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
  Fact 0.0: The Karlsschrein, English name Shrine of Charlemagne, is in Aachen Cathedral.
  Fact 0.1: The Karlsschrein was made in Aachen.
  Fact 0.2: The Karlsschrein was made at the command of Frederick II, Holy Roman Emperor.
  Fact 0.3: The Karlsschrein was completed in 1215.
  Fact 0.4: Frederick II's grandfather was Frederick Barbarossa.
  Fact 0.5: Frederick Barbarossa exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen.
  Fact 0.6: Frederick Barbarossa exhumed Charlemagne's bones in 1165.
  Fact 0.7: The exhumation of Charlemagne's bones happened before the Karlsschrein was completed.
Sentence 1: Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the Holy Roman Emperor from 1155 until his death.
  Fact 1.0: Frederick I's German name is "Friedrich I".
  Fact 1.1: Frederick I's Italian name is "Federico I".
  Fact 1.2: Frederick I lived from 1122 to 10 June 1190.
  Fact 1.3: Frederick I is also known as Frederick Barbarossa.
  Fact 1.4: Frederick Barbarossa's Italian name is "Federico Barbarossa".
  Fact 1.5: Frederick I was the Holy Roman Emperor from 1155 until his death.

Retrieved items:
<item 1>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 1>

<item 2>
Battle of Parma
The Battle of Parma was fought on 18 February 1248 between the forces of Holy Roman Emperor Frederick II and the Guelphs.

 The Guelphs attacked the Imperial camp when Frederick II was away.

 The Imperial forces were defeated and much of Frederick's treasure was lost.
</item 2>

<item 3>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 3>

<item 4>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 4>

<item 5>
List of Holy Roman Empresses
Holy Roman Empress or Empress of the Holy Roman Empire is the title given to the consort (wife) or regent of the Holy Roman Emperor.

 The elective dignity of Holy Roman Emperor was restricted to males only, therefore there was never a Holy Roman Empress regnant, though women such as Theophanu
or Maria Theresa of Austria, who controlled the power of rule, served as de facto Empresses regnant.
</item 5>

<item 6>
Karl Anselm, 4th Prince of Thurn and Taxis
Karl Anselm, 4th Prince of Thurn and Taxis, full German name: "Karl Anselm Fürst von Thurn und Taxis" (2 June 1733, Frankfurt am Main, Free Imperial City of
Frankfurt, Holy Roman Empire – 13 November 1805, Winzer bei Regensburg, Electorate of Bavaria, Holy Roman Empire) was the fourth Prince of Thurn and Taxis,
Postmaster General of the Imperial Reichspost, and Head of the Princely House of Thurn and Taxis from 17 March 1773 until his death on 13 November 1805.

 Karl Anselm served as "Prinzipalkommissar" at the Perpetual Imperial Diet in Regensburg for Joseph II, Holy Roman Emperor and Francis II, Holy Roman Emperor
from 1773 to 1797.
</item 6>

<item 7>
Karlsschrein
The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in
1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
</item 7>

<item 8>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 8>

<item 9>
Frederick I, Holy Roman Emperor
Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the
Holy Roman Emperor from 1155 until his death.

 He was elected King of Germany at Frankfurt on 4 March 1152 and crowned in Aachen on 9 March 1152.

 He became King of Italy in 1155 and was crowned Roman Emperor by Pope Adrian IV on 18 June 1155.

 Two years later, the term "sacrum " ("holy") first appeared in a document in connection with his Empire.

 He was later formally crowned King of Burgundy, at Arles on 30 June 1178.

 He was named "Barbarossa " by the northern Italian cities which he attempted to rule: Barbarossa means "red beard" in Italian; in German, he was known as
"Kaiser Rotbart ", which has the same meaning.
</item 9>

<item 10>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 10>
=== END ITEM id=6df2733499d9629a ===

=== ITEM id=d84be94d7ef2d651 ===
Gold sentences and facts:
Sentence 0:  The name comes from the City of Starachowice, where the factory is located.
  Fact 0.0: The name comes from the City of Starachowice.
  Fact 0.1: The factory is located in the City of Starachowice.
Sentence 1:  It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).
  Fact 1.0: The item was sold in various countries for many years.
  Fact 1.1: The item was sold in countries beyond the Eastern Bloc.
  Fact 1.2: The item was used by the Yemen Army.
Sentence 2: The Eastern Bloc was the group of socialist states of Central and Eastern Europe, generally the Soviet Union and the countries of the Warsaw Pact.
  Fact 2.0: The Eastern Bloc was the group of socialist states of Central and Eastern Europe.
  Fact 2.1: The Eastern Bloc generally included the Soviet Union.
  Fact 2.2: The Eastern Bloc generally included the countries of the Warsaw Pact.

Retrieved items:
<item 1>
J. A. Baczewski
J. A. Baczewski was a Polish company most well known for its fine spirits such as vodka and gin.

 The factory, dating back to the late 18th century, was based in Lwów (Lviv) and until 1939 was one of two most popular Polish export goods.
</item 1>

<item 2>
Shaanxi Automobile Group
Shaanxi Automobile Group Co., Ltd. is a Chinese bus and truck manufacturer headquartered in Xi'an, Shaanxi.

 It employs around 23,000 employees.

 It manufactures heavy duty truck, buses (chassis), medium-size heavy truck and heavy duty axles for trucks.

 It utilizes Magna Steyr and MAN SE technologies.

 Its bus chassis are sold under the Eurostar Bus brand.
</item 2>

<item 3>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 3>

<item 4>
Graph query row:
entity: FSC Star
relationship: LOCATED_IN
related_city: Starachowice
city_name_origin: Starachowice
</item 4>

<item 5>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 5>

<item 6>
FSC Star
Fabryka Samochodów Ciężarowych "Star" was a Polish truck manufacturer.

 The name comes from the City of Starachowice, where the factory is located.

 Their first vehicle was the "Star 20" in 1948.

 The most popular product was the "Star 266".

 The 266 model offered very good quality and powerful engines for a low price.

 It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).

 For many years FSC Star was a state-owned company.

 Star is now owned by MAN AG who eliminated the brand in January 2009.
</item 6>

<item 7>
Volvo FH
The Volvo FH is a heavy truck range produced by Swedish truck manufacturer Volvo Trucks.

 Introduced in late 1993 as FH12 and FH16, production still continues with the now the second generation of FH range model lineup.

 FH stands for Forward control High entry where numbers denominate engine capacity in litres.

 The FH range is one of the most successful truck series ever having sold more than 400,000 units worldwide.
</item 7>

<item 8>
Daimler AG
Daimler AG (] ) is a German multinational automotive corporation.

 Daimler AG is headquartered in Stuttgart, Baden-Württemberg, Germany.

 As of 2014, Daimler owns or has shares in a number of car, bus, truck and motorcycle brands including Mercedes-Benz, Mercedes-AMG, Smart Automobile, Detroit
Diesel, Freightliner, Western Star, Thomas Built Buses, Setra, BharatBenz, Mitsubishi Fuso, MV Agusta as well as shares in Denza, KAMAZ, Beijing Automotive
Group.

 The luxury Maybach brand was terminated at the end of 2012, but revived in April 2015 as "Mercedes-Maybach" versions of the Mercedes-Benz S-Class and G-Class.

 In 2016 Daimler sold 3.0 million vehicles.

 By unit sales, Daimler is the thirteenth-largest car manufacturer and is the largest truck manufacturer in the world.

 In addition to automobiles, Daimler manufactures buses and provides financial services through its Daimler Financial Services arm.

 The company is a component of the Euro Stoxx 50 stock market index.
</item 8>

<item 9>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 9>

<item 10>
Leader Trucks
Leader found their niche in the heavy transport market and built as a rigid 4x4 and 6x6 configuration for tray, tipper and agitator applications, they offered
reliable machines in the industries of earth-moving and off-road construction.

 Leader was notable for being the first manufacturer to fit Caterpillar engines to trucks, to fit automatic transmissions to diesel trucks in assembly, and the
first truck manufacturer in Australia to offer disc brakes.

 Leader boasted its components to be 80 percent Australian built with only the engine, transmission and steering box imported.

 By 1980, Leader had sold its 1000th truck in Australia and had also sold trucks to New Zealand, South Africa, Hong Kong, China and Indonesia.
</item 10>
=== END ITEM id=d84be94d7ef2d651 ===

=== ITEM id=5a6e13414d4d6d7a ===
Gold sentences and facts:
Sentence 0: The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.
  Fact 0.0: The Sports Museum is also known as The Sports Museum of New England.
  Fact 0.1: The Sports Museum is a non-profit museum.
  Fact 0.2: The Sports Museum is currently located in the TD Garden.
  Fact 0.3: The TD Garden is in Boston, Massachusetts.
Sentence 1: TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.
  Fact 1.0: TD Garden is often called Boston Garden.
  Fact 1.1: TD Garden is often called "The Garden".
  Fact 1.2: TD Garden is a multi-purpose arena.
  Fact 1.3: TD Garden is in Boston, Massachusetts.
Sentence 2:  It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
  Fact 2.0: TD Garden opened in 1995.
  Fact 2.1: TD Garden opened as a replacement for the original Boston Garden.
  Fact 2.2: TD Garden has been known as Shawmut Center.
  Fact 2.3: TD Garden has been known as FleetCenter.
  Fact 2.4: TD Garden has been known as TD Banknorth Garden.

Retrieved items:
<item 1>
Sports Legends Museum at Camden Yards
Sports Legends Museum at Camden Yards was a non-profit sports museum in Baltimore, Maryland, United States, owned and operated by the Babe Ruth Birthplace &
Museum.

 It opened on May 14, 2005, with the daughter of celebrated baseball player Babe Ruth in attendance.

 After ten years of operation the museum closed abruptly on October 12, 2015 after failing to reach an agreement with the Maryland Stadium Authority for the
continued use of Camden Station.

 The 22000 sqft museum was adjacent to the main gate of Oriole Park at Camden Yards and has artifacts and interactive exhibits profiling Maryland’s sports
history.

 Exhibits included such area teams as the Baltimore Orioles, Baltimore Ravens, Baltimore Colts, Maryland Terrapins, Baltimore Elite Giants, Baltimore Black Sox,
and the Baltimore Blast.
</item 1>

<item 2>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 2>

<item 3>
Sports Museum of America
The Sports Museum of America (SmA) was the United States' first national sports museum dedicated to the history and cultural significance of sports in America.

 It opened in May 2008 and closed less than nine months later, in February 2009.
</item 3>

<item 4>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 4>

<item 5>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 5>

<item 6>
Papa Murphy's Park
Papa Murphy's Park (formerly known as the Cal Expo Multi-Use Sports Field Facility and Bonney Field) is a sports venue located on the grounds of Cal Expo in
Sacramento, California.

 The soccer-specific stadium has a capacity of 11,569 and includes a full-sized (120 x 80 yard) soccer field.

 Papa Murphy's Park is the current home of Sacramento Republic FC soccer team and former home of PRO Rugby team, Sacramento Express.
</item 6>

<item 7>
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
</item 7>

<item 8>
Estonian Sports Museum
The Estonian Sports and Olympic Museum, founded in 1963 and modernised in 2001, is the largest sports museum in the Baltic states.

 The museum is located on Rüütli street in Tartu, Estonia and the museum has a small outpost in Otepää that concentrates on the Winter sports.

 Before 2016, Estonian Sports and Olympic Museum was named Estonian Sports Museum.
</item 8>

<item 9>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 9>

<item 10>
Fanattic Sports Museum
Fanattic Sports Museum (FSM) is a sports museum, dedicated to various types of sports and located in Ecospace Business Park, New Town, Kolkata, India.

 Established in January 2017, FSM is known as the first sports museum in Kolkata as well as in West Bengal.
</item 10>
=== END ITEM id=5a6e13414d4d6d7a ===

=== ITEM id=5261fdb5e2792e60 ===
Gold sentences and facts:
Sentence 0: Ionolyce is a small genus of butterflies in the family Lycaenidae.
  Fact 0.0: Ionolyce is a small genus of butterflies.
  Fact 0.1: Ionolyce is in the family Lycaenidae.
Sentence 1:  "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.
  Fact 1.0: I. helicon is widespread in the Indoaustralasian region.
  Fact 1.1: I. helicon is widespread from India to the Malay Archipelago.
Sentence 2: India lies on the Indian Plate, the northern portion of the Indo-Australian Plate, whose continental crust forms the Indian subcontinent.
  Fact 2.0: India lies on the Indian Plate.
  Fact 2.1: The Indian Plate is the northern portion of the Indo-Australian Plate.
  Fact 2.2: The continental crust of the Indian Plate forms the Indian subcontinent.
Sentence 3:  It is the seventh-largest country in the world, with a total area of 3287263 km2 .
  Fact 3.0: India is the seventh-largest country in the world.
  Fact 3.1: India has a total area of 3287263 km2.

Retrieved items:
<item 1>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 1>

<item 2>
Genoplesium
Genoplesium commonly known as midge orchids, is a genus of about 50 species of flowering plants in the orchid family, Orchidaceae and is found in Australia, New
Zealand and New Caledonia.

 Midge orchids are terrestrial herbs with a single leaf at the base of the plant.

 They are similar to orchids in the genus "Prasophyllum" in that plants without flowers have a hollow, onion-like leaf.

 The flowers are small but often scented and attractive to their insect pollinators.

 There is disagreement about which species belong to this genus and some taxonomists suggest that most belong in the genus "Corunastylis".
</item 2>

<item 3>
Dregea volubilis
Dregea volubilis is a species of plant in the Apocynaceae family that is native to India.
</item 3>

<item 4>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 4>

<item 5>
Aulacodes traversalis
Aulacodes traversalis is a moth in the Crambidae family.

 It was described by Harrison Gray Dyar Jr. in 1914.

 It is found in Panama.
</item 5>

<item 6>
India
India, officially the Republic of India ("Bhārat Gaṇarājya"), is a country in South Asia.

 It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world.

 It is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast.

 It shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the northeast; and Myanmar (Burma) and Bangladesh to the east.

 In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives.

 India's Andaman and Nicobar Islands share a maritime border with Thailand and Indonesia.
</item 6>

<item 7>
Consul fabius
Consul fabius, the tiger leafwing, is the most common and well known species of the genus "Consul" of subfamily Charaxinae in the brush-footed butterfly family
(Nymphalidae).

 It is found all over the Neotropics.
</item 7>

<item 8>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 8>

<item 9>
Ionolyce
Ionolyce is a small genus of butterflies in the family Lycaenidae.

 "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.

 "I selkon"and "I.brunnescens" are both endemic to the Solomon Islands.
</item 9>

<item 10>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 10>
=== END ITEM id=5261fdb5e2792e60 ===

=== ITEM id=18ff4db77dd7f1c3 ===
Gold sentences and facts:
Sentence 0: Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
  Fact 0.0: Mauritz Stiller was born on 17 July 1883.
  Fact 0.1: Mauritz Stiller died on 18 November 1928.
  Fact 0.2: Mauritz Stiller was a Finnish-Swedish film director.
  Fact 0.3: Mauritz Stiller was best known for discovering Greta Garbo.
  Fact 0.4: Mauritz Stiller brought Greta Garbo to America.
Sentence 1: Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.
  Fact 1.0: Johannes Grenzfurthner was born in 1975 in Vienna.
  Fact 1.1: Johannes Grenzfurthner is an award-winning Austrian artist.
  Fact 1.2: Johannes Grenzfurthner is an Austrian filmmaker.
  Fact 1.3: Johannes Grenzfurthner is an Austrian writer.
  Fact 1.4: Johannes Grenzfurthner is an Austrian actor.
  Fact 1.5: Johannes Grenzfurthner is an Austrian curator.
  Fact 1.6: Johannes Grenzfurthner is an Austrian theatre director.
  Fact 1.7: Johannes Grenzfurthner is an Austrian lecturer.

Retrieved items:
<item 1>
Traceroute (film)
Traceroute is a 2016 Austrian/American documentary film directed by Johannes Grenzfurthner.

 The autobiographical documentary and road movie deals with the history, politics and impact of nerd culture.

 Grenzfurthner calls his film a "personal journey into the uncharted depths of nerd culture, a realm full of dangers, creatures and more or less precarious
working conditions", an attempt to "chase the ghosts of nerddom's past, present and future."

 The film was co-produced by art group monochrom and Reisenbauer Film.

 It features music by Kasson Crooker, Hans Nieswandt, and many others.
</item 1>

<item 2>
Mauritz Stiller
Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
</item 2>

<item 3>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 3>

<item 4>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 4>

<item 5>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 5>

<item 6>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 6>

<item 7>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 7>

<item 8>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 8>

<item 9>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 9>

<item 10>
Johannes Grenzfurthner
Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.

 He is known as the founder, conceptualist and artistic director of monochrom, an international art and theory group.

 Most of his artworks are labelled "monochrom".
</item 10>
=== END ITEM id=18ff4db77dd7f1c3 ===

=== ITEM id=ec155c20ccabdf16 ===
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
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 1>

<item 2>
Managed Money
Managed Money is a 1934 short comedy film directed by Charles Lamont.

 The film stars Frank Coghlan Jr. and Shirley Temple.

 It was also known as "Frolics of Youth" and "Measured Money".

 This was the second film in which Temple starred as Mary Lou.

 The film tells the story of Sonny and Sid prospecting for gold to pay for their military academy education.

 Sonny's sister, Mary Lou stows away in the back of their car.

 Eventually, they meet an inventor who helps them financially.
</item 2>

<item 3>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 3>

<item 4>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 4>

<item 5>
The Muppets (film)
The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.

 The film is directed by James Bobin, written by Jason Segel and Nicholas Stoller, produced by David Hoberman and Todd Lieberman, and stars Segel, Amy Adams,
Chris Cooper and Rashida Jones, as well as Muppet performers Steve Whitmire, Eric Jacobson, Dave Goelz, Bill Barretta, David Rudman, Matt Vogel, and Peter Linz.

 Bret McKenzie served as music supervisor, writing four of the film's five original songs, and Christophe Beck composed the film's score.

 In "The Muppets", devoted fan Walter, his brother Gary, and Gary's girlfriend Mary help Kermit the Frog reunite the disbanded Muppets, as they must raise $10
million to save the Muppet Theater from Tex Richman, a businessman who plans to demolish the studio to drill for oil.
</item 5>

<item 6>
Sam Eagle
Sam Eagle is a Muppet character originating from the television show "The Muppet Show", where he was performed by Frank Oz.

 Sam has appeared in every Muppet film; as himself in "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "Muppets from Space", and "The
Muppets", as well as the Head Schoolmaster in "The Muppet Christmas Carol," Samuel Arrow in "Muppet Treasure Island" and a CIA agent in "Muppets Most Wanted."

 He also appears in the television series, "The Muppets."
</item 6>

<item 7>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 7>

<item 8>
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 8>

<item 9>
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
</item 9>

<item 10>
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 10>
=== END ITEM id=ec155c20ccabdf16 ===
