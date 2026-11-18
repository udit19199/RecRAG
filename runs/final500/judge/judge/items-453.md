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

=== ITEM id=5bc328d1c6b6348f ===
Gold sentences and facts:
Sentence 0: Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.
  Fact 0.0: Neon City is also known as Anno 2053 in Italy.
  Fact 0.1: Neon City is also known as Neonski Grad in Serbia.
  Fact 0.2: Neon City is a 1991 Canadian film.
  Fact 0.3: Neon City is a post-apocalyptic science fiction film.
  Fact 0.4: Neon City was directed by Monte Markham.
  Fact 0.5: Neon City was written by Jeff Begun.
  Fact 0.6: Neon City was written by Ann Lewis Hamilton.
  Fact 0.7: Neon City's writing was credited under the pseudonym Buck Finch.
Sentence 1:  The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
  Fact 1.0: Michael Ironside is in the cast of Neon City.
  Fact 1.1: Vanity is in the cast of Neon City.
  Fact 1.2: Lyle Alzado is in the cast of Neon City.
  Fact 1.3: Valerie Wildman is in the cast of Neon City.
  Fact 1.4: Nick Klar is in the cast of Neon City.
  Fact 1.5: Juliet Landau is in the cast of Neon City.
  Fact 1.6: Richard Sanders is in the cast of Neon City.
  Fact 1.7: Monte Markham is in the cast of Neon City.
Sentence 2: Lyle Martin Alzado (April 3, 1949 – May 14, 1992) was a professional American football defensive end of the National Football League famous for his intense and intimidating style of play.
  Fact 2.0: Lyle Martin Alzado was born on April 3, 1949.
  Fact 2.1: Lyle Martin Alzado died on May 14, 1992.
  Fact 2.2: Lyle Martin Alzado was a professional American football defensive end.
  Fact 2.3: Lyle Martin Alzado played in the National Football League.
  Fact 2.4: Lyle Martin Alzado was famous for his intense and intimidating style of play.

Retrieved items:
<item 1>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 1>

<item 2>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 2>

<item 3>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 3>

<item 4>
Michael Ola
Michael Ola (born April 19, 1988) is an American football offensive tackle for the Buffalo Bills of the National Football League (NFL).

 He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.

 He has also been a member of the Jacksonville Sharks, Montreal Alouettes, Miami Dolphins, Chicago Bears, San Diego Chargers, Detroit Lions, Seattle Seahawks
and New York Giants.
</item 4>

<item 5>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 5>

<item 6>
Time Bombs
Time Bombs is a 2008 Canadian film produced by "Productions de la ruelle".
</item 6>

<item 7>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 7>

<item 8>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 8>

<item 9>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 9>

<item 10>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 10>
=== END ITEM id=5bc328d1c6b6348f ===

=== ITEM id=9414d69e64b2fad7 ===
Gold sentences and facts:
Sentence 0: PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.
  Fact 0.0: PC Magazine is shortened as PCMag.
  Fact 0.1: PC Magazine is an American computer magazine.
  Fact 0.2: PC Magazine is published by Ziff Davis.
Sentence 1:  The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
  Fact 1.0: The magazine focuses on sculpture.
  Fact 1.1: The magazine focuses on sculpture collectors.
  Fact 1.2: The magazine focuses on sculpture enthusiasts.
  Fact 1.3: Painting is also frequently featured in the magazine.
  Fact 1.4: Poetry is also frequently featured in the magazine.

Retrieved items:
<item 1>
Windows: The Official Magazine
Windows: The Official Magazine is a technology magazine produced by Future plc in association with Microsoft, and published worldwide.

 The title describes itself as "A PC magazine for real life", and contains news, features, guides and reviews designed to show readers how to make better use of
their Windows PC.
</item 1>

<item 2>
Computer Press Association
Founded in 1983, the Computer Press Association (CPA) was established to promote excellence in the field of computer journalism.

 The association was composed of working editors, writers, producers, and freelancers who covered issues related to computers and technology.

 The CPA conducted the annual Computer Press Awards, which was the preeminent editorial awards of the computer and technology media.

 The CPA Awards honored outstanding examples in print, broadcast and electronic media.

 Awards were given for print publications, such as PC Magazine; online news media, such as Newsbytes News Network (both were multiple winners); individual
columns and features by well-known journalists such as Steven Levy (author of “”); broadcast awards such as “Best Radio Program”; as well as book awards in
categories such as Best Product Specific Book.

 CPA President Jeff Yablon (1994-1996) developed an updated code of ethics for technology journalists that was adopted by many major trade show groups, most
notably Bruno Blenheim.

 The Computer Press Association disbanded in 2000.
</item 2>

<item 3>
Sculptural Pursuit
Sculptural Pursuit was a quarterly art/literary magazine published by Hammer & Pen Productions, a Denver, Colorado publishing company.

 The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
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
Popular Science
Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on
science and technology subjects.

 "Popular Science" has won over 58 awards, including the American Society of Magazine Editors awards for its journalistic excellence in both 2003 (for General
Excellence) and 2004 (for Best Magazine Section).

 With roots beginning in 1872, "Popular Science" has been translated into over 30 languages and is distributed to at least 45 countries.
</item 5>

<item 6>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 6>

<item 7>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 7>

<item 8>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 8>

<item 9>
Computer Magazine
as well as "Computer Magazine" staff writers, and is currently managed and edited by the industry veteran Christopher Swearingin an MCSE and former CIO and
regarded author as well as contributor/reporter for "Computer Magazine" and other publications.
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
=== END ITEM id=9414d69e64b2fad7 ===

=== ITEM id=2105d7a2663f9bb6 ===
Gold sentences and facts:
Sentence 0: The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver Division, the Shenango Division, and the Conneaut Division.
  Fact 0.0: The Beaver and Erie Canal was also known as the Erie Extension Canal.
  Fact 0.1: The Beaver and Erie Canal was part of the Pennsylvania Canal system.
  Fact 0.2: The Beaver and Erie Canal consisted of three sections.
  Fact 0.3: The three sections of the Beaver and Erie Canal were the Beaver Division, the Shenango Division, and the Conneaut Division.
Sentence 1: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.
  Fact 1.0: The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp.
  Fact 1.1: The Dismal Swamp Canal is located in Virginia and North Carolina.
  Fact 1.2: Virginia and North Carolina are in the United States.

Retrieved items:
<item 1>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 1>

<item 2>
Dismal Swamp State Park
Dismal Swamp State Park is a North Carolina state park in Camden County, North Carolina in the United States.

 The park was created as a state natural area in 1974 with the help of The Nature Conservancy, and on July 28, 2007 the NC General Assembly re-designated it as
a state park.

 It opened to the public in 2008.

 This marked the first time that public access to Great Dismal Swamp was made possible in North Carolina.

 The park covers 14432 acre of protected land on the North Carolina/Virginia border.

 Park offices are three miles (5 km) south of the border on U.S. Route 17 near South Mills.

 Features of the park include the canal which is used regularly by boaters using the Intracoastal Waterway and several miles of hiking and biking trails.
</item 2>

<item 3>
Great Dismal Swamp maroons
The Great Dismal Swamp maroons were freed and escaped slaves who inhabited the marshlands of the Great Dismal Swamp in Virginia and North Carolina.

 Although conditions were harsh, research suggests that thousands lived there between about 1700 and the 1860s.

 Harriett Beecher Stowe told the maroon people's story in her 1856 novel "".

 The most significant research on the settlements began in 2002 with a project by Dan Sayers of American University.
</item 3>

<item 4>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
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
Great Dismal Swamp National Wildlife Refuge
The Great Dismal Swamp National Wildlife Refuge was created in 1974 to help protect and preserve a portion of the Great Dismal Swamp, a marshy region on the
Coastal Plain of southeastern Virginia and northeastern North Carolina between Norfolk, Virginia, and Elizabeth City, North Carolina in the United States.

 It is located in parts of the independent cities of Chesapeake and Suffolk in Virginia, and the counties of Camden, Gates, and Pasquotank in North Carolina.
</item 6>

<item 7>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 7>

<item 8>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 8>

<item 9>
Wallaceton (Chesapeake, Virginia)
Wallaceton is a historic home located at Chesapeake, Virginia.

 The original section was built between 1853 and 1863, as a company store.

 It was expanded after the American Civil War.

 It is a 1 1/2-story, rectangular, Greek Revival style frame dwelling.

 It has flat corner pilasters, a heavy box cornice under the eaves, and a full width front porch.

 Also on the property are a contributing two-room kitchen building and a dairy.

 About 1910, it was relocated approximately 100 feet to the east of the Dismal Swamp Canal to remove it from canal property.

 It was named for John Gallaudet Wallace (1840-1910) a farmer and businessman who fought in the Civil War for the Confederacy, as a Captain of Company C, 61st
Virginia Volunteer Infantry Regiment.
</item 9>

<item 10>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 10>
=== END ITEM id=2105d7a2663f9bb6 ===

=== ITEM id=398273095cd5c2b0 ===
Gold sentences and facts:
Sentence 0: Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.
  Fact 0.0: Ronkonkoma is a major railroad station.
  Fact 0.1: Ronkonkoma is a major transportation hub.
  Fact 0.2: Ronkonkoma is along the Main Line of the Long Island Rail Road.
  Fact 0.3: The Ronkonkoma railroad station is in Ronkonkoma, New York.
Sentence 1: Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.
  Fact 1.0: Ronkonkoma is a hamlet.
  Fact 1.1: Ronkonkoma is a census-designated place (CDP).
  Fact 1.2: Ronkonkoma is on Long Island.
  Fact 1.3: Ronkonkoma is in the Town of Islip.
  Fact 1.4: The Town of Islip is in New York, United States.

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
New Rochelle Harbor (Long Island Sound)
New Rochelle Harbor is the name of a harbor located along Long Island Sound in the city of New Rochelle in Westchester County, New York.

 The Davenport Neck peninsula off the mainland divides New Rochelle's waterfront into two bays; the westerly referred to as New Rochelle Harbor and the easterly
as Echo Bay.
</item 2>

<item 3>
Long Island Electric Railway
The Long Island Electric Railway was a streetcar company operating in Brooklyn, Queens, and Nassau County, New York, United States between 1894 and 1926.

 The company was partially owned by the Long Island Consolidated Electric Companies, a holding company for the Long Island Rail Road and partially by August
Belmont and the Interborough Rapid Transit Company.

 It connected the east end of the Fulton Street El at Crescent Street station in City Line, Brooklyn with Jamaica, Queens, and ran from there to the Nassau
County line at Queens Village and to Far Rockaway, Queens via Nassau County.

 It also had a connection to Belmont Park.

 The New York and Long Island Traction Company used trackage rights over its line from Crescent Street to Queens Village.
</item 3>

<item 4>
Ronkonkoma (LIRR station)
Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.

 The station is the eastern terminus of the Ronkonkoma Branch and the western terminus of the out Branch.
</item 4>

<item 5>
North Creek Railroad Station
The North Creek Railroad Station is a historic railroad station complex located at North Creek, Warren County, New York.

 The complex consists of the railroad station, the freight house, round house, turntable, and horse barn.

 The station was built in 1874 and is a simple, rectangular, gable roofed building with a broad, overhanging strut-supported roof in the Stick-Eastlake style.

 Its exterior is covered with vertical boards.
</item 5>

<item 6>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 6>

<item 7>
Lacona Railroad Station and Depot
Lacona Railroad Station and Depot, also known as New York Central and Hudson River Railroad Station and Depot, is a historic railway depot located at Lacona in
Oswego County, New York.

 It was built in 1891 by the New York Central and Hudson River Railroad.

 It is a small rectangular, one story, gable ended structure.
</item 7>

<item 8>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 8>

<item 9>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 9>

<item 10>
Jamestown Gateway Train Station
The Jamestown Gateway Train Station, also known as the Jamestown Erie Railroad station, and the Erie-Lackawanna Railroad Station, is a historic train station
located at Jamestown in Chautauqua County, New York.

 Although no longer an active railroad station due to a lack of passenger service in the area after a restoration done in 2011 the building currently serves as
a bus transportation center and community space for Jamestown.

 The first train arrived at Jamestown on August 25, 1860 as part of the Atlantic and Great Western Railroad.
</item 10>
=== END ITEM id=398273095cd5c2b0 ===

=== ITEM id=9a622d98fcffd3cd ===
Gold sentences and facts:
Sentence 0:  is a board game created by Glenn Drover and published by Eagle Games in 2003.
  Fact 0.0: The board game in this sentence was created by Glenn Drover.
  Fact 0.1: The board game in this sentence was published by Eagle Games in 2003.
Sentence 1: Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.
  Fact 1.0: Can't Stop is a board game designed by Sid Sackson.
  Fact 1.1: Can't Stop was originally published by Parker Brothers in 1980.
  Fact 1.2: Can't Stop was long out of print in the United States.

Retrieved items:
<item 1>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 1>

<item 2>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 2>

<item 3>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 3>

<item 4>
Plaid Hat Games
Plaid Hat Games is a United States-based board game studio.

 Plaid Hat Games was founded in 2009.

 Board game designer Colby Dauch formed a board game publishing company in order to release the companies first game, Summoner Wars.
</item 4>

<item 5>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 5>

<item 6>
Crash (card game)
Crash is a British card game extension of Nine-card Brag.

 In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
</item 6>

<item 7>
Francis Tresham (game designer)
Francis Tresham is a United Kingdom-based board game designer who has been producing board games since the early 1970s.

 Tresham founded and ran games company Hartland Trefoil (founded 1971), a company well known for its "Civilization" board game, until its sale to MicroProse in
1997.

 His "1829" game was the first of the "18xx" board game series and some of his board games have inspired Sid Meier computer games such as "Railroad Tycoon".
</item 7>

<item 8>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 8>

<item 9>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 9>

<item 10>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 10>
=== END ITEM id=9a622d98fcffd3cd ===

=== ITEM id=1c7cede5730b2632 ===
Gold sentences and facts:
Sentence 0: Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.
  Fact 0.0: Rosario Castellanos Figueroa was born on 25 May 1925.
  Fact 0.1: Rosario Castellanos Figueroa died on 7 August 1974.
  Fact 0.2: Rosario Castellanos Figueroa was a Mexican poet.
  Fact 0.3: Rosario Castellanos Figueroa was a Mexican author.
Sentence 1: Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a series of novels beginning in 1908 with "Anne of Green Gables".
  Fact 1.0: Lucy Maud Montgomery was born on November 30, 1874.
  Fact 1.1: Lucy Maud Montgomery died on April 24, 1942.
  Fact 1.2: Lucy Maud Montgomery published as L.M. Montgomery.
  Fact 1.3: Lucy Maud Montgomery was a Canadian author.
  Fact 1.4: Lucy Maud Montgomery is best known for a series of novels.
  Fact 1.5: The series of novels began in 1908 with Anne of Green Gables.

Retrieved items:
<item 1>
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 1>

<item 2>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 2>

<item 3>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 3>

<item 4>
Anne Shirley
Anne Shirley is a fictional character introduced in the 1908 novel "Anne of Green Gables" by Lucy Maud Montgomery.

 Montgomery wrote in her journal that the idea for Anne's story came from relatives who, planning to adopt an orphaned boy, received a girl instead.

 Anne Shirley's appearance was inspired by a photograph which Montgomery clipped from the Metropolitan Magazine and kept, unaware of the model's identity as the
1900s Gibson Girl Evelyn Nesbit.
</item 4>

<item 5>
Anne of Green Gables
Anne of Green Gables is a 1908 novel by Canadian author Lucy Maud Montgomery (published as L. M. Montgomery).

 Written for all ages, it has been considered a children's novel since the mid-twentieth century.

 It recounts the adventures of Anne Shirley, an 11-year-old orphan girl who is mistakenly sent to Matthew and Marilla Cuthbert, a middle-aged brother and sister
who had intended to adopt a boy to help them on their farm in the fictional town of Avonlea on Prince Edward Island.

 The novel recounts how Anne makes her way with the Cuthberts, in school, and within the town.
</item 5>

<item 6>
L.M. Montgomery's Anne of Green Gables
L.M. Montgomery's Anne of Green Gables is a Canadian television film based on Lucy Maud Montgomery's novel of the same name.

 It first aired on YTV on February 15, 2016 and starred Ella Ballentine, Martin Sheen and Sara Botsford.

 Montgomery's granddaughter, Kate Macdonald Butler, was one of the film's executive producers.

 The film's world premiere was held February 2, 2016 at the Canadian Museum of History.
</item 6>

<item 7>
Anne of Ingleside
Anne of Ingleside is a children's novel by Canadian author Lucy Maud Montgomery.

 It was first published in July 1939 by McClelland and Stewart (Toronto) and the Frederick A. Stokes Company (New York).

 It is the tenth of eleven books that feature the character of Anne Shirley, and Montgomery's final published novel.

 (Two novels that occur later in the "Anne" chronology were actually published years earlier.

 As well, the short story collection "The Blythes Are Quoted", written in 1941/42, but not published until 2009, concludes the Anne chronology.)
</item 7>

<item 8>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 8>

<item 9>
Emily's Quest
Emily's Quest is a novel and the last of the "Emily" trilogy by Lucy Maud Montgomery.

 After finishing "Emily Climbs", Montgomery suspended writing "Emily's Quest" and published "The Blue Castle"; she resumed writing and published in 1927.
</item 9>

<item 10>
Rosario Castellanos
Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.

 Along with the other members of the Generation of 1950 (the poets who wrote following the Second World War, influenced by César Vallejo and others), she was
one of Mexico's most important literary voices in the last century.

 Throughout her life, she wrote eloquently about issues of cultural and gender oppression, and her work has influenced feminist theory and cultural studies.

 Though she died young, she opened the door of Mexican literature to women, and left a legacy that still resonates today.
</item 10>
=== END ITEM id=1c7cede5730b2632 ===

=== ITEM id=b6de00916a03b051 ===
Gold sentences and facts:
Sentence 0:  The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
  Fact 0.0: The women's events include the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park.
  Fact 0.1: The women's events include the Roxy Pro at Sunset Beach.
  Fact 0.2: The women's events include the Billabong Pro Maui at Honolua Bay, Maui.
Sentence 1: Haleʻ iwa is a North Shore community and census-designated place (CDP) in the Waialua District of the island of Oʻ ahu, City and County of Honolulu.
  Fact 1.0: Haleʻiwa is a North Shore community.
  Fact 1.1: Haleʻiwa is a census-designated place (CDP).
  Fact 1.2: Haleʻiwa is in the Waialua District of the island of Oʻahu.
  Fact 1.3: Haleʻiwa is in the City and County of Honolulu.

Retrieved items:
<item 1>
Triple Crown of Surfing
The Vans Triple Crown of Surfing is a Hawaiian specialty series of professional surfing events, offering three events to men and three events to women.

 For the men, those events are the Reef Hawaiian Pro at Haleiwa Ali'i Beach Park; the O'Neill World Cup of Surfing at Sunset Beach; and the Billabong Pipeline
Masters at the Banzai Pipeline.

 The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
</item 1>

<item 2>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 2>

<item 3>
Andy Irons
Philip Andrew Irons (July 24, 1978November 2, 2010) was an American professional surfer.

 Irons learned to surf on the dangerous and shallow reefs of the North Shore in Kauai, Hawaii.

 Over the course of his professional career, he won three world titles (2002, 2003, 2004), three Quiksilver Pro France titles (2003, 2004, 2005), two Rip Curl
Pro Search titles (2006 and 2007) and 20 elite tour victories including the Vans Triple Crown of Surfing four times from 2002-2006.

 On September 3, 2010 he won the Billabong Pro Teahupoo in Tahiti.

 He and his family hosted the Annual Irons Brothers Pinetrees Classic, a contest for youngsters.

 The Governor of Hawaii declared February 13 forever "Andy Irons Day".

 He is the only surfer to have won a title at every venue on the ASP calendar.
</item 3>

<item 4>
Santa Barbara International Marathon
The Santa Barbara International Marathon and Half Marathon is an annual race that takes place in Goleta, California and Santa Barbara, California.

 The inaugural race was on December 6, 2009.

 For the last five years the Santa Barbara City College has been the official host for the finish line and general operations.

 Rusty and June Snow are the co-directors of the race with Dan Campbell as the technical director.

 The Snow's moved from the East Coast and wanted to bring a marathon to Santa Barbara.

 After a few years of getting the race approved and organized it was finally set to happen in 2009, Rusty Snow states, “I learned to be stubborn from running
marathons.”

 The Santa Barbara International Marathon course serves as a Boston Marathon qualifier.
</item 4>

<item 5>
Ocean City Open
The Ocean City Open is a defunct tennis tournament that was played on the Grand Prix tennis circuit in 1977.

 The event was held in Ocean City, Maryland.

 Vitas Gerulaitis won the singles title while Alex Metreveli and Bill Scanlon partnered to win the doubles title.
</item 5>

<item 6>
Reef Hawaiian Pro
The Reef Hawaiian Pro is a surfing event held annually on the North Shore, Oahu, Hawaii.

 It is currently the first leg of the Triple Crown of Surfing.

 Surfers compete for a share of $135,000 (men) or $35,000 (women) prize money.

 Since it carries a 6-star rating, the event is critical for those who want to qualify for the ASP World Tour.
</item 6>

<item 7>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 7>

<item 8>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 8>

<item 9>
2012 Monterrey Open
The 2012 Monterrey Open was a women's tennis tournament played on outdoor hard courts.

 It was the 4th edition of the Monterrey Open and was an International tournament on the 2012 WTA Tour.

 It took place at the Sierra Madre Tennis Club in Monterrey, Mexico, from 20 to 26 February.

 Tímea Babos won the singles title.
</item 9>

<item 10>
World Match Racing Tour
The World Match Racing Tour is an annual match racing tour and World Championship in the match racing event of sailing.

 The World Match Racing Tour has "special event"-status from the International Sailing Federation.

 The Tour awards over US$1.5 million in prize money, with points awarded at each event culminating in the crowning of the ISAF Match Racing World Champion.

 A US$500,000 overall prize pool for the Championship is now on offer.
</item 10>
=== END ITEM id=b6de00916a03b051 ===

=== ITEM id=6713b67fc76c2e83 ===
Gold sentences and facts:
Sentence 0: Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.
  Fact 0.0: Paola Suárez was born on 23 June 1976.
  Fact 0.1: Paola Suárez is a retired professional tennis player.
  Fact 0.2: Paola Suárez is from Argentina.
Sentence 1:  She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.
  Fact 1.0: Paola Suárez was one of the most prominent women's doubles players throughout the early and mid-2000s.
  Fact 1.1: Paola Suárez won eight Grand Slam titles.
  Fact 1.2: All eight of Paola Suárez's Grand Slam titles were won with Virginia Ruano Pascual.
  Fact 1.3: Paola Suárez held the No. 1 doubles ranking for 87 non-consecutive weeks.
Sentence 2:  He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.
  Fact 2.0: Paola Suárez reached a career high singles ranking of 274 in July 2011.
  Fact 2.1: Paola Suárez is much better known as a leading doubles player with a huge serve.

Retrieved items:
<item 1>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 1>

<item 2>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 2>

<item 3>
1999 Italian Open – Women's Doubles
The 1999 Italian Open – Doubles was the doubles event of the fifty-fifth edition of the tennis tournament played at Rome, Italy, the most prestigious tennis
tournament in Southern Europe.

 It was the fifth WTA Tier I tournament of the year, and part of the European claycourt season.

 Virginia Ruano Pascual and Paola Suárez were the defending champions, but lost to Martina Hingis and Anna Kournikova in the quarterfinals.
</item 3>

<item 4>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 4>

<item 5>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 5>

<item 6>
Paola Suárez
Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.

 She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano
Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.

 She was also a singles semifinalist at the 2004 French Open.
</item 6>

<item 7>
Clarisa Fernández
Clarisa Fernández (born 28 August 1981) is a retired Argentine tennis player who is best known for her semi-final appearance at the 2002 French Open, the first
Argentine since Sabatini in 1992 to reach that stage.

 Fernández was ranked 87th in the world at the time of her shocking result.

 She defeated, in order: (Q) Lubomira Bacheva, Jelena Kostanić, Kim Clijsters, (13) Elena Dementieva, Paola Suárez, before losing to second-seeded Venus
Williams in straight sets.
</item 7>

<item 8>
1999 Copa Colsanitas – Singles
The 1999 Copa Colsanitas – Singles was the singles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Paola Suárez was the defending champion but lost in the semifinals to Christína Papadáki.
</item 8>

<item 9>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 9>

<item 10>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 10>
=== END ITEM id=6713b67fc76c2e83 ===

=== ITEM id=00530ca43c28db06 ===
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
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 1>

<item 2>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 2>

<item 3>
Graph query row:
museum: The Harris Museum, Art Gallery & Preston Free Public Library
location_relationship: LOCATED_IN
located_in: Preston
person: None
person_relationship: None
</item 3>

<item 4>
Edmund Robert Harris
Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris
Institute or Art School, Harris Technical School and the Harris Orphanage.
</item 4>

<item 5>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 5>

<item 6>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
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
Hanbys Corner, Delaware
The last Hanby to occupy the property, Albert T. Hanby (1881-1947), another son of Samuel, attended West Chester State College before getting his law degree
from Penn Law School.

 Albert became a Philadelphia lawyer and left his farm at Hanby's Corner to be used for the good of "all the children in Delaware".

 He and his wife created a foundation in 1945 to protect the property from further development.

 Today the YMCA operates their Hanby Camp there, and the trust provide scholarships for kids who might not otherwise be able to attend.

 In 2007 that support was over $70,000.
</item 8>

<item 9>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 9>

<item 10>
Edmund de Unger
Edmund Robert Anthony de Unger (Hungarian: "Odon Antal Robert de Unger" , b 6 August 1918, Budapest - d 25 January 2011, Ham, Surrey) was a Hungarian-born
property developer and art collector.

 In London he built up the Keir Collection, one of the greatest post-war collections of Islamic art, bequeathed in 2008 to the Pergamon Museum of Islamic Art in
Berlin.

 The arrangement for the museum to curate the collection came to an end in July 2012.

 The collection is now hosted by the Dallas Museum of Art as of May 2014 for a 15-year renewable loan.
</item 10>
=== END ITEM id=00530ca43c28db06 ===

=== ITEM id=50d893cd38606387 ===
Gold sentences and facts:
Sentence 0:  The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
  Fact 0.0: The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part.
  Fact 0.1: Sammy Hagar played the rhythm guitar part on a Gibson acoustic.
Sentence 1: OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist Sammy Hagar.
  Fact 1.0: OU812 is pronounced 'Oh You Ate One Too'.
  Fact 1.1: OU812 is the eighth studio album by American hard rock band Van Halen.
  Fact 1.2: OU812 was released in 1988.
  Fact 1.3: OU812 is the second Van Halen album to feature vocalist Sammy Hagar.

Retrieved items:
<item 1>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 1>

<item 2>
Finish What Ya Started
"Finish What Ya Started" is a song by Van Halen taken from their 1988 album "OU812".

 Despite the album being seemingly complete, Eddie Van Halen came up with the riff at 2 in the morning and went down to his then-neighbor Sammy Hagar to show
it.

 Hagar let Eddie in, and the two played guitars in his balcony until they had a completed song.

 Once Eddie left, Hagar decided to write the lyrics despite being late at night.

 The theme wound up being unfulfilled sex, summed up by Hagar as "blue balls.

 In the song, Eddie Van Halen recorded his guitar part on a Fender Stratocaster plugged direct into the studio mixing console.

 The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
</item 2>

<item 3>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 3>

<item 4>
Keisuke Kuwata (album)
Keisuke Kuwata is the eponymous studio album recorded by Keisuke Kuwata, a frontman of the Japanese rock band Southern All Stars.

 It was released by Taishita label under the Victor Entertainment in July 1988, shortly after the 10th anniversary of the band.

 Aside from the 1982 live recording album "Kamon Yuzo and Victor Wheels Live" released under the pseudonym Yūzō Kamon, it was first Kuwata's solo effort.
</item 4>

<item 5>
OU812
OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist
Sammy Hagar.

 Van Halen started work on the album in September 1987 and completed it in April 1988, just one month before its release.
</item 5>

<item 6>
The Year of the Voyager
The Year Of The Voyager is a double DVD/CD released by Nevermore.

 It was released in Europe on October 20, 2008 and in North America on November 25, 2008 via Century Media.

 The set covers the "This Godless Endeavor" touring cycle, starting with live footage from the U.S. Gigantour 2005, the Metal Mania festival 2006 in Poland, the
Wacken Open Air festival in 2006 in Germany and the main DVD show recorded at the Zeche in Bochum, Germany.

 Bonus material includes two songs from Century Media USA 10th Anniversary Party 2001, all promo videos and an interview with singer Warrel Dane which was
recorded at the Roax Film Studios in Berlin in the spring of 2008.
</item 6>

<item 7>
&quot;Happy&quot; in Galoshes
"Happy" in Galoshes is the second solo album by American rock singer Scott Weiland.

 Weiland, known for his roles as the lead singer in Stone Temple Pilots and Velvet Revolver, released his first album, "12 Bar Blues" in 1998.

 Ten years later, ""Happy" in Galoshes" served as the official follow-up.

 Two versions have been released, a single disc and double disc deluxe version—the deluxe edition offers a second disc of 10 extra tracks.

 The album was released November 25, 2008 on Weiland's own Softdrive Records.

 Produced by Doug Grean and Scott Weiland, with select tracks recorded by Steve Albini, the album features guest appearances by Paul Oakenfold and No Doubt
members Adrian Young, Tony Kanal, and Tom Dumont.

 Oakenfold appears on a cover of David Bowie's "Fame."

 Weiland has cited Bowie as one of his main influences.
</item 7>

<item 8>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 8>

<item 9>
Hoarse &amp; Roaring
Hoarse & Roaring is the debut full-length album by neofolk band Parlor Hawk, released in 2010 on Northplatte Records.

 The album was produced by Joshua James and featured Neon Trees bassist Branden Campbell on each track (except "Saddest Song") as well as Fictionist member
Stuart Maxfield on the tracks "14 Years" and "Home".

 It was mixed by Todd Burke (Ben Harper, Jack Johnson) and mastered by Reuben Cohen (Edward Sharpe & The Magnetic Zeros, Bruno Mars).
</item 9>

<item 10>
OU812 Tour
The OU812 Tour was a concert tour by hard rock band Van Halen.

 It was the second tour to include dates in Japan, and was the second with Sammy Hagar as vocalist (who had settled into the role more by this point).

 It followed the Monsters of Rock Tour 1988, which had formed the first part of the promotion for the "OU812" album.
</item 10>
=== END ITEM id=50d893cd38606387 ===

=== ITEM id=85aa1b853a68c6d4 ===
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
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 1>

<item 2>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 2>

<item 3>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 3>

<item 4>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 4>

<item 5>
Robert Holdstock
Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish
fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
</item 5>

<item 6>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 6>

<item 7>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 7>

<item 8>
Robert Holdstock bibliography
This is a bibliography of fantasy author Robert Holdstock.
</item 8>

<item 9>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 9>

<item 10>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 10>
=== END ITEM id=85aa1b853a68c6d4 ===
