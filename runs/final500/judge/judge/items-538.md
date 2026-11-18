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

=== ITEM id=c3cc0859cb88e8af ===
Gold sentences and facts:
Sentence 0:  Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college levels.
  Fact 0.0: The institution has three divisions.
  Fact 0.1: Through its three divisions, the institution offers a wide variety of liberal arts courses.
  Fact 0.2: Through its three divisions, the institution offers a wide variety of science courses.
  Fact 0.3: The courses are offered at the secondary level.
  Fact 0.4: The courses are offered at the British A-level level.
  Fact 0.5: The courses are offered at the United States junior college level.
Sentence 1:  Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in education.
  Fact 1.0: Grammar was a core liberal art.
  Fact 1.1: Logic was a core liberal art.
  Fact 1.2: Rhetoric was a core liberal art.
  Fact 1.3: Arithmetic also played a somewhat lesser part in education.
  Fact 1.4: Geometry also played a somewhat lesser part in education.
  Fact 1.5: The theory of music also played a somewhat lesser part in education.
  Fact 1.6: Astronomy also played a somewhat lesser part in education.

Retrieved items:
<item 1>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 1>

<item 2>
Santa Barbara City College
Santa Barbara City College (SBCC) is a two-year community college that opened in 1909 and is located on a 74 acre campus in the city of Santa Barbara,
California, USA.

 SBCC was named co-winner of the 2013 Aspen Institute Prize for Community College Excellence, earning it a #1 national ranking.

 The college was recognized for its quality and focus in four areas: facilitating underrepresented and minority student success, student learning outcomes,
degree completion and transfer rates, and labor market success in securing good jobs after college.

 Santa Barbara City College was established by the Santa Barbara High School District in 1909, making it one of the oldest community colleges in California.

 The college was discontinued shortly after World War I, and its work largely taken over by the Santa Barbara State Normal School, which became the Santa
Barbara State College, and later, the University of California, Santa Barbara.
</item 2>

<item 3>
St. Joseph's Intermediate and Commercial School
St. Joseph's Intermediate and Commercial School (St. Joseph's Commercial School, SJICS, or St. Joe's Sherbourne) is a former Catholic High school located in the
Cabbagetown neighbourhood of Toronto, Ontario, Canada part of the Metropolitan Separate School Board (now the Toronto Catholic District School Board).

 The school was later renamed to Thomas Merton Academy (Thomas Merton, TMA, or Merton) in 1985 after the Anglo-American monk Thomas Merton.
</item 3>

<item 4>
List of St. John's Seminary (California) people
The list of St. John's Seminary (California) people is a compilation of lists of notable alumni, faculty, and current students of St. John's Seminary in
Camarillo, California, United States.

 St. John's Seminary grants graduate degrees for seminarians preparing for the priesthood, as well as a graduate degree for lay persons interested in pastoral
ministry.

 The St. John's Seminary College was the undergraduate division of the seminary before it closed in the early 21st century.

 The table of notable alumni lists the date of graduation from St. John's college, seminary, or both, if applicable.

 It is not unusual for seminarians to have received their undergraduate education at a different institution than their seminary training.
</item 4>

<item 5>
Aalborghus Gymnasium
Aalborghus Gymnasium is an upper secondary school in the city of Aalborg, in North Jutland in Denmark.

 It offers both the traditional three-year program and also the two-year Higher Preparatory Examination (HF) program.

 The subjects taught at the school range from Religion and Music to Spanish and Natural Geography.

 Aalborghus Gymnasium attempts to focus on the musical and creative side of students as well as taking an international perspective on issues.

 Students begin their studies at Aalborghus by selecting a stream of studies.

 Each stream has two or three subjects that are the focus of the studies.

 English/Social Studies is one example of a stream a student can select.
</item 5>

<item 6>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 6>

<item 7>
St. John's College, Belize
St. John's College has three divisions, and a number of central academic centres and activities.

 Through its three divisions, it offers a wide variety of liberal arts and science courses at the secondary, British A-level, and United States junior college
levels.

 St. John's College is a Roman Catholic institution in the Jesuit tradition, one of the oldest, largest, and most diverse educational institutions in Belize,
founded by the Jesuits in 1887.
</item 7>

<item 8>
St. John's High School (South Carolina)
St. John's High School (SJHS) is a senior high school on Johns Island, South Carolina.

 It is a part of the Charleston County School District.

 St. John's is home to approximately 300 students and 30 faculty and staff.

 St. John's school mascot is The Mighty Islanders, sporting royal blue and maroon as the school colors.

 St. John's offers Advanced Placement and dual credit courses totaling at over 30 hours of offered college credit, as well as 3 career academies in Hospitality
and Tourism, Computer Science, and Culinary Arts.

 St. John's competes at the A level in football, volleyball, basketball (boys and girls), wrestling, soccer, track, baseball, and softball.

 The Islanders also offer marching band, agriculture and green house, weightlifting, competitive academic team, and student council.
</item 8>

<item 9>
Liberal arts education
The liberal arts (Latin: "artes liberales") are those subjects or skills that in classical antiquity were considered essential for a free person (Latin:
"liberalis", "worthy of a free person") to know in order to take an active part in civic life, something that (for Ancient Greece) included participating in
public debate, defending oneself in court, serving on juries, and most importantly, military service.

 Grammar, logic, and rhetoric were the core liberal arts, while arithmetic, geometry, the theory of music, and astronomy also played a (somewhat lesser) part in
education.
</item 9>

<item 10>
St. John's College High School, Belize
St. John's College High School is a high school for boys situated in Belize City, Belize.

 It was founded in 1887.

 The High School exists to educate academically talented young men in a Jesuit environment of self-discipline, love of learning, and service to others.

 The school 's curriculum is complemented by sports and extracurricular activities.

 The third and fourth form classes follow the Caribbean Secondary Education Certificate (CSEC) curricula and sit the regionally administered examinations at the
end of their fourth year.
</item 10>
=== END ITEM id=c3cc0859cb88e8af ===

=== ITEM id=5c77f9fa78ab2e97 ===
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
Village School (Great Neck, New York)
Village School (also known as VS) is an American long-established public alternative school.

 It is a member of Coalition of Essential Schools and is located in the village of Great Neck, New York, serving students in grades 8 through 12.

 Village School is one of three high schools in the Great Neck School District, which includes Great Neck North High School and Great Neck South High School.

 Village School offers its 39 students an outdoor education program, college preparatory program, and inclusion of students with disabilities.
</item 1>

<item 2>
East Lyme High School
East Lyme High School is a high school located in the Flanders Village region of East Lyme, Connecticut.

 It is operated by East Lyme Public Schools.

 The mascot is Sven the Viking.

 Students from the town of Salem, Connecticut in grades 9 through 12 attend high school in East Lyme (as they have no high school of their own); this will be
the case until at least 2016 when the current co-op agreement between the two towns expires.
</item 2>

<item 3>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
</item 3>

<item 4>
Saddle Rock Elementary School
Saddle Rock Elementary School or simply "Saddle Rock" is an elementary school, comprising grades Kindergarten through 5.

 It is a public school located in Great Neck, New York, USA as part of the Great Neck School District.

 The school has been recognized by the Blue Ribbon Schools Program.
</item 4>

<item 5>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 5>

<item 6>
New Rochelle Harbor (Long Island Sound)
New Rochelle Harbor is the name of a harbor located along Long Island Sound in the city of New Rochelle in Westchester County, New York.

 The Davenport Neck peninsula off the mainland divides New Rochelle's waterfront into two bays; the westerly referred to as New Rochelle Harbor and the easterly
as Echo Bay.
</item 6>

<item 7>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 7>

<item 8>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 8>

<item 9>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 9>

<item 10>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 10>
=== END ITEM id=5c77f9fa78ab2e97 ===

=== ITEM id=7993def92db39963 ===
Gold sentences and facts:
Sentence 0: David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
  Fact 0.0: David Ross Lederman was born December 12, 1894.
  Fact 0.1: David Ross Lederman died August 24, 1972.
  Fact 0.2: David Ross Lederman was an American film director.
  Fact 0.3: David Ross Lederman was noted for his Western, action, and adventure films of the 1930s and 1940s.
Sentence 1: Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
  Fact 1.0: Stefan Ruzowitzky was born 25 December 1961.
  Fact 1.1: Stefan Ruzowitzky is an Austrian film director.
  Fact 1.2: Stefan Ruzowitzky is an Austrian screenwriter.

Retrieved items:
<item 1>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 1>

<item 2>
Otto Struve
Otto Struve (August 12, 1897 – April 6, 1963) was a astronomer.

 In Russian, his name is sometimes given as Otto Lyudvigovich Struve (Отто Людвигович Струве); however, he spent most of his life and his entire scientific
career in the United States.

 Otto was the descendant of famous astronomers of the Struve family; he was the son of Ludwig Struve, grandson of Otto Wilhelm von Struve and great-grandson of
Friedrich Georg Wilhelm von Struve.

 He was also the nephew of Karl Hermann Struve.
</item 2>

<item 3>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 3>

<item 4>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 4>

<item 5>
Ruggero Deodato
Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
</item 5>

<item 6>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 6>

<item 7>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 7>

<item 8>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 8>

<item 9>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 9>

<item 10>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 10>
=== END ITEM id=7993def92db39963 ===

=== ITEM id=142d9c5693d041c9 ===
Gold sentences and facts:
Sentence 0: The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.
  Fact 0.0: The Dundee Canal was an industrial canal.
  Fact 0.1: The Dundee Canal was in Clifton and Passaic.
  Fact 0.2: Clifton and Passaic are in Passaic County, New Jersey.
Sentence 1:  It was built between 1858 and 1861 and ran parallel to the Passaic River.
  Fact 1.0: The Dundee Canal was built between 1858 and 1861.
  Fact 1.1: The Dundee Canal ran parallel to the Passaic River.
Sentence 2:  It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.
  Fact 2.0: It runs parallel to the coast.
  Fact 2.1: It is separated from the Atlantic Ocean by barrier beaches.
  Fact 2.2: The barrier beaches include Palm Beach Island.
Sentence 3:  The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
  Fact 3.0: The lagoon is connected to the Atlantic Ocean.
  Fact 3.1: The lagoon is connected to the Atlantic Ocean by two inlets.
  Fact 3.2: The two inlets are permanent.
  Fact 3.3: The two inlets are man-made.

Retrieved items:
<item 1>
Lowell Power Canal System and Pawtucket Gatehouse
The Lowell Power Canal System is the largest power canal system in the United States, at 5.6 miles in length.

 The system's estimated output is 10,000 horsepower, operating six major canals on two levels, controlled by numerous gates.

 The system was begun in the 1790s, beginning its life as a transportation canal called the Pawtucket Canal, which was constructed to get logs from New
Hampshire down the Merrimack River to shipbuilding centers at Newburyport, Massachusetts, bypassing the 30-plus-foot drop of the Pawtucket Falls.
</item 1>

<item 2>
Caisson lock
A caisson lock was invented in the late 18th century as a solution to minimise the use of great volumes of water required to raise and lower canal boats through
large height differences.

 It was normal to only raise and lower boats through small height differences of a few feet when traversing undulating terrain.

 A solution was required when either large height differences were encountered or water was in short supply.

 The Caisson (or Caisoon) was thought to be one solution.

 The technology of the time was not capable of achieving this type of construction economically with current building materials.

 This design attempted was a type of canal lock in which a narrowboat is floated into a sealed watertight box and raised or lowered between two different canal
water levels.

 It was designed primarily as a water-saving measure, and also was an attempt to minimise construction costs compared with other engineering solutions of the
time.

 In use it was capable of replacing up to seven conventional locks.

 Other design benefits were speed of boat descent/ascent, and only a little loss of water when operating compared with a conventional boat lock.
</item 2>

<item 3>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 3>

<item 4>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
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
South Lake Worth Inlet
The South Lake Worth Inlet, also known as the Boynton Inlet, is an artificial cut through a barrier beach connecting the south end of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 The inlet is 130 ft wide and 6 ft and 12 ft deep.
</item 6>

<item 7>
Lake Worth Inlet
The Palm Beach Inlet, also known as the Lake Worth Inlet is an artificial cut through a barrier island connecting the northern part of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 It is bordered by the town of Palm Beach on the south, and by the town of Palm Beach Shores to the north.

 The inlet is also the entrance channel for the Port of Palm Beach.

 Its coordinates are .
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
Lake Worth Lagoon
The Lake Worth Lagoon is a lagoon located in Palm Beach County, Florida.

 It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.

 The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
</item 9>

<item 10>
Edison Sault Power Canal
The Edison Sault Power Canal supplies the Saint Marys Falls Hydropower Plant, a Cloverland Electric Cooperative hydroelectric plant, in Sault Ste. Marie,
Michigan.

 Excavation of the power canal began in September 1898 and was completed in June 1902.

 The canal and hydroelectric complex were named a Historic Civil Engineering Landmark in 1983.
</item 10>
=== END ITEM id=142d9c5693d041c9 ===

=== ITEM id=6063bcf232480560 ===
Gold sentences and facts:
Sentence 0: Rynella is an unincorporated community in Iberia Parish, Louisiana, that consists of a few residential dwellings and a volunteer fire department along Highway 329 (also known as the Avery Island Highway).
  Fact 0.0: Rynella is an unincorporated community.
  Fact 0.1: Rynella is in Iberia Parish, Louisiana.
  Fact 0.2: Rynella consists of a few residential dwellings.
  Fact 0.3: Rynella has a volunteer fire department.
  Fact 0.4: The residential dwellings and the volunteer fire department are along Highway 329.
  Fact 0.5: Highway 329 is also known as the Avery Island Highway.
Sentence 1:  The name "Rynella" allegedly is an acronym deriving from the last two letters of the first names of three local women, Rosema"RY", Pauli"NE", and Lei"LA" McIlhenny (someone eventually adding an extra L to the resulting word).
  Fact 1.0: The name Rynella allegedly is an acronym.
  Fact 1.1: The acronym derives from the last two letters of the first names of three local women.
  Fact 1.2: The three local women are Rosema, Pauli, and Lei McIlhenny.
  Fact 1.3: The acronym uses the letters RY from Rosema, NE from Pauli, and LA from Lei.
  Fact 1.4: Someone eventually added an extra L to the resulting word.
Sentence 2:  They were daughters of Edward Avery McIlhenny, a noted Louisiana conservationist who presided over McIlhenny Company, maker of Tabasco brand pepper sauce at nearby Avery Island.
  Fact 2.0: The three women were daughters of Edward Avery McIlhenny.
  Fact 2.1: Edward Avery McIlhenny was a noted Louisiana conservationist.
  Fact 2.2: Edward Avery McIlhenny presided over McIlhenny Company.
  Fact 2.3: McIlhenny Company is the maker of Tabasco brand pepper sauce.
  Fact 2.4: McIlhenny Company is at nearby Avery Island.
Sentence 3: Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers ("Capsicum frutescens" var.
  Fact 3.0: Tabasco sauce is a brand of hot sauce.
  Fact 3.1: Tabasco sauce is made exclusively from tabasco peppers.
  Fact 3.2: The tabasco pepper is Capsicum frutescens var. tabasco.
Sentence 4:  "tabasco"), vinegar, and salt.
  Fact 4.0: Tabasco sauce is made with vinegar.
  Fact 4.1: Tabasco sauce is made with salt.

Retrieved items:
<item 1>
Rynella, Louisiana
Rynella is an unincorporated community in Iberia Parish, Louisiana, that consists of a few residential dwellings and a volunteer fire department along Highway
329 (also known as the Avery Island Highway).

 The name "Rynella" allegedly is an acronym deriving from the last two letters of the first names of three local women, Rosema"RY", Pauli"NE", and Lei"LA"
McIlhenny (someone eventually adding an extra L to the resulting word).

 They were daughters of Edward Avery McIlhenny, a noted Louisiana conservationist who presided over McIlhenny Company, maker of Tabasco brand pepper sauce at
nearby Avery Island.

 (At one time McIlhenny operated a general store at Rynella and he owned much of the land that now comprises and surrounds the community.)
</item 1>

<item 2>
Tabasco sauce
Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers ("Capsicum frutescens" var.

 "tabasco"), vinegar, and salt.

 It is produced by McIlhenny Company of Avery Island, Louisiana.
</item 2>

<item 3>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 3>

<item 4>
Louisiana Hot Sauce
Louisiana Hot Sauce is a brand of hot sauce manufactured in New Iberia, Louisiana by The Original Louisiana Hot Sauce Co., which is owned by Southeastern Mills
Inc.

 The product's labeling includes the word "original", and it is sometimes referred to as "Original Louisiana Hot Sauce" and "Original Louisiana Brand Hot
Sauce."

 It is a common hot sauce in the U.S. state of Louisiana.

 Bruce Foods was the previous owner and manufacturer of the brand, and sold it to Southeastern Mills Inc. in April 2015.
</item 4>

<item 5>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 5>

<item 6>
Riverton, New York
Riverton, New York was a planned community built in 1973 in the town of Henrietta, New York, along the Genesee River.

 Riverton was seventh of the thirteen communities receiving Federal guarantee assistance from HUD as part of their New Communities Program created by the Urban
Growth and New Community Development Act of 1970.

 $12 million was earmarked for the community.

 Riverton was supposed to encompass a 2335 acre land area bordered on the north by the New York State Thruway, on the west by the Genesee River, on the east by
East River Rd., and on the south by the Rush-Henrietta town line, with some development continuing east along Erie Station Rd.

 Original plans called for construction to occur for a 16-year period and eventually house over 25,000 people, with the community including an 18-hole golf
course, 12 swimming pools, 3 artificial lakes, a marina on the Genesee, and commercial and industrial centers.

 However, only a small portion of this was built along Scottsville-West Henrietta Rd.
</item 6>

<item 7>
Rachel, Nevada
Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.

 As of the 2010 census it had a population of 54.

 As the closest habitation to the Nellis Air Force Range and Area 51, Rachel enjoys a modest celebrity, particularly among aviation enthusiasts and UFO hunters.

 North of the town is the Quinn Canyon Range, which has the ghost town of Adaven.
</item 7>

<item 8>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 8>

<item 9>
Valentina (hot sauce)
Valentina is a brand of pourable hot sauce manufactured by Salsa Tamazula, a company in Guadalajara, Mexico.

 It is typically sold in 12.5-ounce and very large (one-liter or 34-ounce) glass bottles, with a flip-top cap permanently attached to the bottle (the cap does
not unscrew).

 The sauce, and the parent company's Tamazula hot sauce, are made with puya chilis from Jalisco state – similar to the Guajillo chili and known by the name
"guajillo puya" – is described as thicker than Tabasco sauce and less vinegary, with more chile flavor.

 Valentina comes in two varieties, hot and extra hot, and the sauce is known for its use on tacos and its taste, as opposed to only for its heat.

 Valentina's ingredients are water, chili peppers, vinegar, salt, spices and sodium benzoate (as a preservative).
</item 9>

<item 10>
Sriracha sauce
Sriracha (Thai: ศรีราชา ,  ] ; ) is a type of hot sauce or chili sauce made from a paste of chili peppers, distilled vinegar, garlic, sugar, and salt.

 It is named after the coastal city of Si Racha, in Chonburi Province of eastern Thailand, where it may have been first produced for dishes served at local
seafood restaurants.
</item 10>
=== END ITEM id=6063bcf232480560 ===

=== ITEM id=6b54c60fbdd0d62c ===
Gold sentences and facts:
Sentence 0: Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with punctate stigmas on the female flowers.
  Fact 0.0: Lithocarpus is a genus in the beech family Fagaceae.
  Fact 0.1: Lithocarpus differs from "Quercus" in the erect spikes of insect-pollinated male flowers.
  Fact 0.2: Lithocarpus differs from "Quercus" in the short styles with punctate stigmas on the female flowers.
Sentence 1:  It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.
  Fact 1.0: Lithocarpus contains 17 species of shrubs and small trees.
  Fact 1.1: The Lithocarpus species are native from southern Florida to Mexico and South America.

Retrieved items:
<item 1>
Lithocarpus glaber
Lithocarpus glaber, the Japanese oak, is a tree species in the genus "Lithocarpus" found in Japan, China and Taïwan.
</item 1>

<item 2>
Lepidozamia comprises two Australian rainforest cycads (2n=18). Orange Field, a Sri Lankan tea factory founded in 1950, makes about 500,000 kg of black tea
annually for Middle Eastern and CIS markets, sourcing from 750 smallholders. Lithocarpus has 334 accepted East/Southeast Asian species; three Java endemics
listed here are endangered, while L. glaber occurs in Japan, China and Taiwan. Oaks (Querc
</item 2>

<item 3>
Duranta erecta
Duranta erecta is a species of flowering shrub in the verbena family Verbenaceae, native from Mexico to South America and the Caribbean.

 It is widely cultivated as an ornamental plant in tropical and subtropical gardens throughout the world, and has become naturalized in many places.

 It is considered an invasive species in Australia, China, South Africa and on several Pacific Islands.

 Common names include golden dewdrop, pigeon berry, and skyflower.

 In Mexico, the native Nahuatl name for the plant is xcambocoché.

 In Tonga it is known as mavaetangi (tears of departure).
</item 3>

<item 4>
Lithocarpus platycarpus
Lithocarpus platycarpus is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.

.
</item 4>

<item 5>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 5>

<item 6>
Lithocarpus crassinervius
Lithocarpus crassinervius is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 6>

<item 7>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 7>

<item 8>
Lithocarpus kostermansii
Lithocarpus kostermansii is a species of plant in the Fagaceae family.

 It is a tree endemic to Java in Indonesia.

 It is an endangered species threatened by habitat loss.
</item 8>

<item 9>
The entries describe diverse plants and one crab genus. Mascarene Dictyosperma album, the wind-shedding “hurricane palm,” is widely cultivated but nearly
extinct in its native range. Canary Islands Adenocarpus viscosus grows above 1,800 m. Dregea vines occur in Africa and southern Asia; Duranta shrubs and trees
are ornamental, with D. erecta invasive in several regions. Tropical Dieffenbachia are popular houseplants
</item 9>

<item 10>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 10>
=== END ITEM id=6b54c60fbdd0d62c ===

=== ITEM id=3e8c4b222845d2b8 ===
Gold sentences and facts:
Sentence 0:  His primary jockey is Tony McCoy.
  Fact 0.0: The unnamed subject of this sentence has Tony McCoy as its primary jockey.
  Fact 0.1: Tony McCoy is a primary jockey.
Sentence 1:  Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.
  Fact 1.0: Tony McCoy is based in Ireland and the UK.
  Fact 1.1: Tony McCoy rode a record 4,358 winners.
  Fact 1.2: Tony McCoy was Champion Jockey a record 20 consecutive times.
  Fact 1.3: Tony McCoy was Champion Jockey every year he was a professional.

Retrieved items:
<item 1>
Irish flat racing Champion Jockey
The Champion Jockey of flat racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1950.
</item 1>

<item 2>
Gordon Richards
Sir Gordon Richards (5 May 1904 – 10 November 1986) was an English jockey.

 He was the British flat racing Champion Jockey 26 times and is often considered the world's greatest ever jockey.

 He remains the only flat jockey to have been knighted.
</item 2>

<item 3>
Fred Winter
Frederick Thomas Winter CBE (20 September 1926 – 5 April 2004) was a British National Hunt racing racehorse jockey and trainer.

 He was British jump racing Champion Jockey four times and British jump racing Champion Trainer eight times.

 He is the only person to have won the Cheltenham Gold Cup, Champion Hurdle and Grand National as both jockey and trainer.

 Winter won the Grand National four times, as a jockey in 1957 (Sundew) and 1962 (Kilmore), and as a trainer in 1965 (Jay Trump) and 1966 (Anglo).
</item 3>

<item 4>
Irish jump racing Champion Jockey
The Champion Jockey of National Hunt racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1947.

 The current champion, as of 2017, is Ruby Walsh.
</item 4>

<item 5>
British flat racing Champion Jockey
The Champion Jockey of flat racing in Great Britain is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey and the number of winners for each year since 1840.

 The seasonal record of jockeys' winners was published for the first time in 1846.

 The championship was sponsored for the first time in 2009 by online casino 32Red and is currently sponsored by Stobart Group.

 The 2016 title was determined by the number of winners ridden between 30 April and 15 October, in both turf and all-weather racing.

 Prior to 2015 the title was decided over the full length of the Flat turf season, from March to November, and carried no prize money to the winner.

 A prize of £25,000 to the champion jockey was also introduced as part of the 2015 changes.
</item 5>

<item 6>
Ernest Piggott
Ernest "Ernie" Piggott (1878–1967) born Nantwich, Cheshire, England was a leading British jump racing jockey, whose family has become one of the leading
dynasties in British horseracing.

 He was three times Champion Jockey and three times Grand National winner.

 His son, (Ernest) Keith Piggott (1904–1993), was also a leading jump jockey and National-winning trainer, while his grandson is the 11-times British flat
racing Champion Jockey, Lester Piggott.
</item 6>

<item 7>
Tony McCoy
Sir Anthony Peter McCoy, OBE (born 4 May 1974), commonly known as A. P. McCoy or Tony McCoy, is a Northern Irish former horse racing jockey.

 Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.

 He stands 1.78 m (5'10"), far taller than most jockeys.
</item 7>

<item 8>
German flat racing Champion Jockey
The Champion Jockey of flat racing in Germany is the jockey who has ridden the most winning horses during a season.

 The list of the Champion Jockey started 1870.

 The list below shows the Champion Jockey for each year since 1984.
</item 8>

<item 9>
Bart Cummings
James Bartholomew Cummings {'1': ", '2': ", '3': ", '4': "} (14 November 1927 – 30 August 2015), also known by his initials J. B. Cummings, was one of the most
successful Australian racehorse trainers.

 He was known as the "Cups King", referring to the Melbourne Cup, as he won 'the race that stops a nation' a record twelve times.

 During his lifetime Cummings was considered an Australian cultural icon and an Australian National Living Treasure.
</item 9>

<item 10>
Binocular (horse)
Binocular, foaled on March 17, 2004 in France, is a French thoroughbred retired racehorse, born to sire, Enrique and dam Bleu Ciel Et Blanc.

 He is owned by J. P. McManus and trained by Nicky Henderson.

 His primary jockey is Tony McCoy.
</item 10>
=== END ITEM id=3e8c4b222845d2b8 ===

=== ITEM id=f0d6b34ba58fca7e ===
Gold sentences and facts:
Sentence 0:  The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
  Fact 0.0: The film concentrated on Ellsworth "Bumpy" Johnson.
  Fact 0.1: Laurence Fishburne played Ellsworth "Bumpy" Johnson in the film.
  Fact 0.2: The film concentrated on Dutch Schultz.
  Fact 0.3: Tim Roth played Dutch Schultz in the film.
  Fact 0.4: The film concentrated on Lucky Luciano.
  Fact 0.5: Andy García played Lucky Luciano in the film.
Sentence 1:  Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.
  Fact 1.0: Lucky Luciano is considered the father of modern organized crime in the United States.
  Fact 1.1: Lucky Luciano is considered the father of modern organized crime in the United States for establishing the first Commission.
  Fact 1.2: Lucky Luciano established the first Commission.

Retrieved items:
<item 1>
Illtown
Dante and his girlfriend Micky run a very profitable drug operation in a seaside town, aided and abetted by a host of teens who sell the smack at discos around
town, as well as by Lucas, a corrupt cop who's on the take.

 Their downfall comes when they suspect one of the boys, Pep, of ripping them off, and his accidental death causes disloyalty among the teens, who suspect Dante
offed them.

 All of this is perfect for the return of Gabriel, a one-time partner of Dante, who has just been released from jail, and has an almost angelic demeanor and the
certainty that he can fix everyone's lives.
</item 1>

<item 2>
Frank Colacurcio
Francis Colacurcio, Sr. (June 18, 1917 – July 2, 2010) was an American businessman and boss of the Seattle crime family known for running strip clubs in
Seattle, Washington.

 He gained notoriety as a subject of ongoing federal investigations into organized crime in the city and was suspected of being an organized crime boss.
</item 2>

<item 3>
James Marcello
James J. Marcello (born December 13, 1943), also known variously as Little Jimmy, Jimmy Light and as Jimmy the Man Marcello, is an imprisoned crime boss who was
a front boss for the Chicago Outfit criminal organization in the 1980s, 1990s and early 2000s.

 Organized crime observers identified Marcello as a figurehead during that period while the organization's day-to-day operations actually were run by John “No
Nose” DiFronzo, Joseph “Joey the Clown” Lombardo, Joseph “Joe the Builder” Andriacchi and Angelo J. LaPietra.

 Marcello is the Cousin of Michael Hayes Marcello, and a cousin of Robert Urbinatti.

 He is also the father of Rocco Marcello.
</item 3>

<item 4>
The Godfather (soundtrack)
The Godfather is the soundtrack from the film of the same name, released in 1972 by Paramount Records, and in 1991 on compact disc by MCA.

 Unless noted, the cues were composed by Nino Rota and conducted by Carlo Savina (who was credited on the LP, but not the CD).

 The song "I Have but One Heart" is sung by Al Martino, who performed it in the film as character Johnny Fontane.
</item 4>

<item 5>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 5>

<item 6>
Hoodlum (film)
Hoodlum is a 1997 American crime drama film that gives a fictionalized account of the gang war between the Italian/Jewish mafia alliance and the Black gangsters
of Harlem that took place in the late 1920s and early 1930s.

 The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
</item 6>

<item 7>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 7>

<item 8>
Ang Utol Kong Hoodlum
Ang Utol Kong Hoodlum (lit.

 "My Brother is a Hoodlum") is a Filipino drama series developed for TV5 created by Deo J. Fajardo.

 It stars JC de Vera and Jasmine Curtis-Smith.

 It is a remake of the original movie where Robin Padilla played the role of Ben, a hoodlum, and Vina Morales, as Bing.

 The movie was first released in 1991, then a sequel entitled "Miss na Miss Kita: Ang Utol Kong Hoodlum 2" was made the following year.

 It is produced by Vic Del Rosario Jr., and Manuel V. Pangilinan and it also marks as the first primetime series produced by Viva Television for TV5 after a
decade.
</item 8>

<item 9>
Joseph Todaro Jr.
Joseph "Big Joe" Todaro Jr. (born 1945 or 1946) is a Buffalo, New York businessman and former organized crime figure involved in labor racketeering,
loansharking, illegal gambling, narcotics, and murder for hire.

 Joe Todaro Jr. became a business agent for the Laborers' International Union of North America (LIUNA) Local 210.

 In 1984, Joe Todaro Jr. allegedly became the underboss of the Buffalo crime family after his father became the new head of the crime family, upon the
retirement of his predecessor Samuel "Sam the Farmer" Frangiamore.

 In 1990, Joe Todaro Jr. resigned as business agent following investigations on the local's alleged ties to organized crime.

 Outside of organized crime, Todaro operates La Nova Pizzeria, a popular pizza restaurant in Buffalo.
</item 9>

<item 10>
Hoodlum (soundtrack)
Hoodlum is the soundtrack to the 1997 American crime drama film "Hoodlum".

 It was released on August 12, 1997 by Interscope Records and consisted of a blend of hip hop and R&B music.

 The soundtrack peaked at 94 on the "Billboard" 200 and 23 on the Top R&B Albums and contained Mobb Deep's single "Hoodlum" which went to 29 on the Hot Rap
Singles.
</item 10>
=== END ITEM id=f0d6b34ba58fca7e ===

=== ITEM id=15aa74fe58f54d47 ===
Gold sentences and facts:
Sentence 0: Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual art movement known as pop art.
  Fact 0.0: Andy Warhol was born Andrew Warhola.
  Fact 0.1: Andy Warhol's life span runs from August 6, 1928 to February 22, 1987.
  Fact 0.2: Andy Warhol was an American artist.
  Fact 0.3: Andy Warhol was an American director.
  Fact 0.4: Andy Warhol was an American producer.
  Fact 0.5: Andy Warhol was a leading figure in the visual art movement known as pop art.
Sentence 1: Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's production company.
  Fact 1.0: Alexander Butler was a British film director.
  Fact 1.1: Alexander Butler made over sixty features and short films.
  Fact 1.2: Alexander Butler made those features and short films during the 1910s and 1920s.
  Fact 1.3: Alexander Butler made many of those features and short films for G.B. Samuelson's production company.

Retrieved items:
<item 1>
The Andy Warhol Story
The Andy Warhol Story (1966) is an underground film directed by Andy Warhol with cinematography by Paul Morrissey, and starring Edie Sedgwick and Rene Ricard
(as Andy Warhol).
</item 1>

<item 2>
Beautiful Darling
Beautiful Darling: The Life and Times of Candy Darling, Andy Warhol Superstar is a 2010 feature-length documentary film about Candy Darling, the transgender
pioneer, actress and Andy Warhol Superstar.

 The film was written and directed by James Rasin and features Chloë Sevigny as "the voice of Candy Darling", reading from Candy's private diaries and letters.

 Patton Oswalt voices Andy Warhol and Truman Capote.

 Louis Durra composed the score.
</item 2>

<item 3>
John Warhola
John Warhola (May 31, 1925 – December 24, 2010) played a pivotal role in maintaining the legacy of his younger brother, pop artist Andy Warhol, assigned
responsibility by their father on his deathbed to ensure that Andy attended college and serving as a trustee of the Andy Warhol Foundation for the Visual Arts
after his brother's death in 1987.

 Warhola oversaw the establishment of The Andy Warhol Museum in Pittsburgh and the Andy Warhol Museum of Modern Art in Medzilaborce, Slovakia.
</item 3>

<item 4>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 4>

<item 5>
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 5>

<item 6>
Green Coca-Cola Bottles
Green Coca-Cola Bottles is a 1962 painting by Andy Warhol that depicts numerous Coca-Cola bottles.

 Andy Warhol’s painting “Green Coca-Cola Bottles” attempted to take a mainstream item and converted it into a piece of art.

 Warhol’s piece was a hybrid craft that is to say that he utilized a silkscreen technique, which mechanicalized some aspects of the painting, but also kept it
so that he had to put in some of his individualized “unevenness” across the painting, to reel in the human aspect.

 The painting engenders an optimistic message for the American public, which is best described in Warhol’s own words, “What’s grand about this country is that
America started the tradition where the richest consumers buy essentially the same thing as the poorest... you can know that the President drinks Coke, Liz
Taylor drinks Coke, and, just think, you can drink Coke, too.

 A Coke is a Coke, and no amount of money can get you a better Coke.”

 Here Warhol goes to show that the democratic equality, which allowed the birth of a uniformly governed consumerist mindset, has given them an equal satiating
medium.
</item 6>

<item 7>
Green Coca-Cola Bottles
With respect to its artistic merits, Warhol’s painting utilizes repetitive imagery in the shape of hundred and twelve almost identical bottles, evoking a sense
of mechanicalization which overlaps his former usage of mass culture objects.

 The widespread display of advertisement flooded the American public during the sixties, and Warhol had been successful in mapping it onto his canvas, no wonder
it is considered one of Warhol’s masterpieces.
</item 7>

<item 8>
Andy Warhol
Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual
art movement known as pop art.

 His works explore the relationship between artistic expression, celebrity culture, and advertising that flourished by the 1960s, and span a variety of media,
including painting, silkscreening, photography, film, and sculpture.

 Some of his best known works include the silkscreen paintings "Campbell's Soup Cans" (1962) and "Marilyn Diptych" (1962), the experimental film "Chelsea Girls"
(1966), and the multimedia events known as the "Exploding Plastic Inevitable" (1966–67).
</item 8>

<item 9>
Exploding Plastic Inevitable
The Exploding Plastic Inevitable, sometimes simply called Plastic Inevitable or EPI, was a series of multimedia events organized by Andy Warhol between 1966 and
1967, featuring musical performances by The Velvet Underground and Nico, screenings of Warhol's films, and dancing and performances by regulars of Warhol's
Factory, especially Mary Woronov and Gerard Malanga.

 "Andy Warhol's Exploding Plastic Inevitable" is also the title of an 18-minute film by Ronald Nameth with recordings from one week of performances of the shows
which were filmed in Chicago, Illinois, in 1966.

 In December 1966 Warhol included a one-off magazine called "The Plastic Exploding Inevitable" as part of the "Aspen" No. 3 package.
</item 9>

<item 10>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 10>
=== END ITEM id=15aa74fe58f54d47 ===

=== ITEM id=74cc8a722237fbd5 ===
Gold sentences and facts:
Sentence 0: The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US skyscraper.
  Fact 0.0: The Woolworth Building is at 233 Broadway, Manhattan, New York City.
  Fact 0.1: The Woolworth Building was designed by architect Cass Gilbert.
  Fact 0.2: The Woolworth Building was constructed between 1910 and 1912.
  Fact 0.3: The Woolworth Building is an early US skyscraper.
Sentence 1: 1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.
  Fact 1.0: 1 New York Plaza is an office building in New York City's Financial District.
  Fact 1.1: 1 New York Plaza was built in 1969.
  Fact 1.2: 1 New York Plaza is at the intersection of South and Whitehall Streets.

Retrieved items:
<item 1>
F. W. Woolworth Building (Watertown, New York)
The Woolworth Building is an historic building in Watertown, New York.

 It is a contributing building in the Public Square Historic District.

 Plans for the Woolworth Building were begun in 1916 by Frank W. Woolworth, the founder of the Woolworth's chain of department stores.
</item 1>

<item 2>
Woolworth Building
The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US
skyscraper.

 The original site for the building was purchased by F. W. Woolworth and his real estate agent Edward J. Hogan by April 15, 1910, from the Trenor Luther Park
Estate and other owners for $1.65 million.

 By January 18, 1911, Woolworth and Hogan had acquired the final site for the project, totaling $4.5 million.

 More than a century after its construction, it remains, at 241.4 m , one of the 100 tallest buildings in the United States as well as one of the 30 tallest
buildings in New York City.

 It has been a National Historic Landmark since 1966, and a New York City landmark since 1983.
</item 2>

<item 3>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 3>

<item 4>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
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
Westchester Avenue (NYW&amp;B station)
Westchester Avenue is a former railroad station located in the borough of the Bronx in New York City, partially suspended over Amtrak's busy Northeast Corridor
line.

 It was built in 1908 with rich terra cotta detailing to a design by Cass Gilbert, who would later employ similar terra cotta detailing in his 1910 design for
the Woolworth Building.

 Train service to the station ceased in 1937, and as of 2014 the station was a ruin in poor condition.
</item 6>

<item 7>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 7>

<item 8>
Ware's Department Store
Ware’s Department Store is a historic building located in the city of New Rochelle in Westchester County, New York and is significant for both architectural and
commercial reasons.

 Ware's was Westchester’s first and, for many years, largest department store, and was prominently located on New Rochelle’s fashionable Main Street.

 The store operated from 1881 to the late 1930s, when the property was sold to the retailer Bloomingdale's to serve as their first suburban department store
location.

 Not only was Ware's a key place in New Rochelle during its 20th-century boom years, its founder and owner, Howard R. Ware was a leading figure in the rapidly
growing community as well.

 Ware first moved to New Rochelle from Massachusetts at the age of 13 and began to work as a clerk.

 In 1881 he became partner in the firm of Ware & Sheffield, which eventually became a stock company in 1913.

 renamed He was a director and vice president of the National City Bank of New Rochelle, a founder and first president of the local Y.M.C.A. from 1899 to 1916,
and an active member of St.
</item 8>

<item 9>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 9>

<item 10>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 10>
=== END ITEM id=74cc8a722237fbd5 ===

=== ITEM id=5e689d8c28529939 ===
Gold sentences and facts:
Sentence 0: William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.
  Fact 0.0: William Dwight Schultz was born on November 24, 1947.
  Fact 0.1: William Dwight Schultz is an American actor.
  Fact 0.2: William Dwight Schultz is an American voice artist.
Sentence 1:  He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the Squirrel in "CatDog".
  Fact 1.0: William Dwight Schultz is well known in animation as the mad scientist Dr. Animo in the "Ben 10" series.
  Fact 1.1: William Dwight Schultz is well known in animation as Chef Mung Daal in the children's cartoon "Chowder".
  Fact 1.2: William Dwight Schultz is well known in animation as Eddie the Squirrel in "CatDog".
Sentence 2: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 2.0: Chowder is an American animated television series.
  Fact 2.1: Chowder was created by C. H. Greenblatt.
  Fact 2.2: Chowder was created for Cartoon Network.

Retrieved items:
<item 1>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 1>

<item 2>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 2>

<item 3>
Graph query row:
character: Mung Daal
character_show_relationship: PART_OF
creative_work: Chowder
voice_relationship: None
voice_actor: William Dwight Schultz
</item 3>

<item 4>
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
</item 4>

<item 5>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 5>

<item 6>
Rodger Bumpass
Rodger Bumpass (born November 20, 1951) is an American actor and voice actor.

 He is best known for his long-running role as Squidward Tentacles on the American animated television series "SpongeBob SquarePants".

 He voices many other characters on the show, including Dr. Forrest and various anchovies.

 He also voiced The Chief in the animated series "Where on Earth Is Carmen Sandiego?

", and Mr. Besser, the school principal in the animated series "The Kids from Room 402".

 Bumpass has many other credits in animated films, animated television series, and video games.
</item 6>

<item 7>
Alan Young
Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television
host/personality who "TV Guide" called "The Charlie Chaplin of Television".

 He was best known for his role as naive Wilbur Post in the television comedy series "Mister Ed" (1961–1966).

 Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983)
and in various other films, TV series and video games until his death.

 During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy
Awards in 1951.

 He also appeared in a number of feature films, starting from 1946, including the 1960 film "The Time Machine" and from the 1980s gaining a new generation of
viewers appearing in numerous Walt Disney Productions films as both an actor and voice actor.
</item 7>

<item 8>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 8>

<item 9>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 9>

<item 10>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 10>
=== END ITEM id=5e689d8c28529939 ===

=== ITEM id=c2d9911027f7a136 ===
Gold sentences and facts:
Sentence 0:  It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
  Fact 0.0: The area occupies 40 km (15 sq mi).
  Fact 0.1: The area extends into each of the six municipalities on the island.
Sentence 1: Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the fifth largest in Spain's national parks system.
  Fact 1.0: Sierra de Guadarrama National Park is a national park in Spain.
  Fact 1.1: The Spanish name of Sierra de Guadarrama National Park is Parque Nacional de la Sierra de Guadarrama.
  Fact 1.2: Sierra de Guadarrama National Park covers nearly 34,000 hectares.
  Fact 1.3: Sierra de Guadarrama National Park is the fifth largest in Spain's national parks system.

Retrieved items:
<item 1>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 1>

<item 2>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 2>

<item 3>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 3>

<item 4>
Golf La Moraleja
It has 6,000 shareholder members.

 In 2013 it was voted the best golf course in Spain by Deporte & Business and in 2014, La Moraleja 3 was voted the second best golf course in Spain by the
American magazine Golf Digest.

 The Club has hosted competitions such as the World Cup of Golf, the Spanish Open or the Paddle Tennis World Championship.
</item 4>

<item 5>
Landes de Gascogne Regional Natural Park
Landes de Gascogne Regional Natural Park (French: "Parc naturel régional des Landes de Gascogne") is a protected area of pine forest, wetland and oceanic
coastline located in the Aquitaine region of southwestern France.
</item 5>

<item 6>
Golf La Moraleja
Golf La Moraleja is one of Spain’s most important clubs.

 It is the largest Spanish golf club for its number of its holes and the largest in Europe located in a metropolitan area.

 Founded in 1973, it currently has four 18-hole golf courses, all rated par 72, designed by Jack Nicklaus.

 It also contains a 9-hole short course, 8 tennis courts, 12 paddle tennis courts, 2 squash courts, one covered and three open-air swimming pools, a gym and a
spa.

 Added to these are two clubhouses, a children’s chalet and a tennis clubhouse (on 1 and 2 courses), plus a prebuilt modular clubhouse for courses 3 and 4.

 Courses 1 and 2 are located at La Moraleja residential estate (Alcobendas, Madrid).

 Courses 3 and 4 are located in the municipality of Algete.

 It has 6,000 shareholder members.
</item 6>

<item 7>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 7>

<item 8>
Garajonay National Park, on La Gomera in Spain, covers 40 km², spans six municipalities, and became a national park in 1981 and UNESCO World Heritage Site in
1986. Bulgaria’s Pirin National Park covers 403.56 km², reaches 2,914 m at Vihren, and has been protected since 1962 and UNESCO-listed since 1983. Spain’s
Guadarrama National Park protects nearly 34,000
</item 8>

<item 9>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 9>

<item 10>
Hermigua, on northeastern La Gomera, lies 12 km from the island capital; Garajonay National Park covers its south. Tenerife, the Canaries’ largest and most
populous island, contains Teide National Park and the emblematic Roque Cinchado. Guanche mummies preserve pre-Spanish indigenous remains; a Guanche pagan
organization was founded in La Laguna in 2001. Chile’s Cay is an apparently inactive stratovolcano; Penco
</item 10>
=== END ITEM id=c2d9911027f7a136 ===
