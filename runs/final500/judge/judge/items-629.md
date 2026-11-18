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

=== ITEM id=aed4d7e3a5264a47 ===
Gold sentences and facts:
Sentence 0:  It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
  Fact 0.0: The work is the third installment of the 'Naked' series.
  Fact 0.1: 'Naked Killer' is a work in the 'Naked' series released in 1992.
  Fact 0.2: 'Naked Weapon' is a work in the 'Naked' series released in 2002.
Sentence 1: Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
  Fact 1.0: Naked Weapon is also titled 赤裸特工.
  Fact 1.1: Naked Weapon is a 2002 film.
  Fact 1.2: Naked Weapon is a Hong Kong film.
  Fact 1.3: Naked Weapon is an action-thriller film.
  Fact 1.4: Naked Weapon was directed by action choreographer Tony Ching.
  Fact 1.5: Naked Weapon stars Maggie Q.
  Fact 1.6: Naked Weapon stars Anya Wu.
  Fact 1.7: Naked Weapon stars Daniel Wu.

Retrieved items:
<item 1>
Naked Weapon
Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
</item 1>

<item 2>
Naked Gun 33⅓: The Final Insult
Naked Gun 33⅓: The Final Insult is a 1994 comedy film, and the third and final installment in "The Naked Gun" film series, which was based on the "Police Squad!

" television series.
</item 2>

<item 3>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 3>

<item 4>
American Pie Presents: The Naked Mile
American Pie Presents: The Naked Mile (also known as American Pie: The Naked Mile) is a 2006 American sex comedy film released by Universal Pictures.

 It is the second installment in the "American Pie Presents" series and the fifth installment in the "American Pie" franchise.

 The film begins a story arc that concludes with "" (2007).

 John White stars as Erik Stifler, a high school senior who is given a "guilt free pass" by his girlfriend, Tracy Sterling (Jessy Schram), and so visits the
Beta House fraternity led by his cousin, Dwight Stifler (Steve Talley), to run a mile naked.

 Christopher McDonald co-stars as Erik's father, Harry, and Eugene Levy once again plays Jim's Dad, who turns out to be a family friend of both Erik's and
Tracy's.

 Also, it is in this film that his name is revealed to be "Noah Levenstein".
</item 4>

<item 5>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 5>

<item 6>
Naked Soldier
Naked Soldier (絕色武器) is a 2012 Hong Kong action film directed by Marco Mak and starring Jennifer Tse.

 It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
</item 6>

<item 7>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 7>

<item 8>
The entries span film, television, documentaries, and film distribution: Philippine series Sine Novela adapts Viva classics; The Doom Generation (1995) is Gregg
Araki’s violent “Teenage Apocalypse” film; Naked Gun 33⅓ concludes its comedy trilogy, while Naked City (1958–63) is a semi-documentary police drama. Night Life
is a zombie comedy; The Runaways portrays the band’s formation and Currie-Jett relationship. Consolidated Film Industries was
</item 8>

<item 9>
The Naked Gun 2½: The Smell of Fear
The Naked Gun 2½: The Smell of Fear is a 1991 comedy film.

 It is the sequel to the 1988 film "" and the second installment in "The Naked Gun" film series.

 The film stars Leslie Nielsen as the comically bumbling Police Lt. Frank Drebin of "Police Squad!

".

 Priscilla Presley plays the role of Jane, with O. J. Simpson as Nordberg and George Kennedy as police captain Ed Hocken.

 The film also features Robert Goulet (who previously made a "special guest star" appearance on "Police Squad!")

 as the villainous Quentin Hapsburg and Richard Griffiths as renewable fuel advocate Dr. Albert S. Meinheimer (as well as his evil double, Earl Hacker).

 Zsa Zsa Gabor, Mel Tormé and members of the Chicago Bears have cameo roles.
</item 9>

<item 10>
Mel Gibson filmography
Five years later he played Martin Riggs in the buddy cop action comedy "Lethal Weapon" alongside Danny Glovera role he later reprised in its sequels "Lethal
Weapon 2" (1989), "Lethal Weapon 3" (1992), and "Lethal Weapon 4" (1998).
</item 10>
=== END ITEM id=aed4d7e3a5264a47 ===

=== ITEM id=8e7d380b58b829f6 ===
Gold sentences and facts:
Sentence 0: 888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.
  Fact 0.0: 888 7th Avenue is a 628 ft (191 m) tall modern-style office skyscraper.
  Fact 0.1: 888 7th Avenue is in Midtown Manhattan.
  Fact 0.2: 888 7th Avenue was completed in 1969.
  Fact 0.3: 888 7th Avenue has 46 floors.
Sentence 1: 200 West Street is the global headquarters of the Goldman Sachs investment banking firm.
  Fact 1.0: 200 West Street is the global headquarters of the Goldman Sachs investment banking firm.
Sentence 2:  The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.
  Fact 2.0: The building is 749 ft tall.
  Fact 2.1: The building has 44 stories.
  Fact 2.2: The building is located on West Street.
  Fact 2.3: The building is located between Vesey Street and Murray Street.
  Fact 2.4: The building is in Lower Manhattan.

Retrieved items:
<item 1>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 1>

<item 2>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 2>

<item 3>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 3>

<item 4>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 4>

<item 5>
Robert B. Atwood Building
The Robert B. Atwood Building is an office building located at 550 West 7th Avenue in Downtown Anchorage, Alaska.

 The building houses government offices for the State of Alaska.

 Standing at 20 stories and 81 m (265 ft), it is the second-tallest building in Alaska.

 The building was formerly known as the "Bank of America Center".

 Together with the slightly taller Conoco-Phillips Building, the skyscraper defines Anchorage's skyline.
</item 5>

<item 6>
Woolworth Building
The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US
skyscraper.

 The original site for the building was purchased by F. W. Woolworth and his real estate agent Edward J. Hogan by April 15, 1910, from the Trenor Luther Park
Estate and other owners for $1.65 million.

 By January 18, 1911, Woolworth and Hogan had acquired the final site for the project, totaling $4.5 million.

 More than a century after its construction, it remains, at 241.4 m , one of the 100 tallest buildings in the United States as well as one of the 30 tallest
buildings in New York City.

 It has been a National Historic Landmark since 1966, and a New York City landmark since 1983.
</item 6>

<item 7>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 7>

<item 8>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 8>

<item 9>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 9>

<item 10>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 10>
=== END ITEM id=8e7d380b58b829f6 ===

=== ITEM id=1eef963914613555 ===
Gold sentences and facts:
Sentence 0: Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).
  Fact 0.0: Groom Lake is a salt flat in Nevada.
  Fact 0.1: Groom Lake is used for runways of the Nellis Bombing Range Test Site airport.
  Fact 0.2: The Nellis Bombing Range Test Site airport has the code KXTA.
Sentence 1:  Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
  Fact 1.0: Groom Lake is located within the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.1: Groom Lake Valley is the namesake of the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.2: Groom Lake is 25 miles south of Rachel, Nevada.
Sentence 2: Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.
  Fact 2.0: Rachel is a census-designated place (CDP).
  Fact 2.1: Rachel is in Lincoln County, Nevada, United States.

Retrieved items:
<item 1>
Papoose Lake
Papoose Lake is a dry lake bed located in Lincoln County, Nevada, USA.

 The lake lies within the plot of land referred to as the Groom Lake facility (aka Area 51) and is a restricted area.

 The lake is located a few miles southwest of the Groom lake facility nestled adjacent the Papoose Range.
</item 1>

<item 2>
John Skillern House
The John Skillern House is a historic cabin located 25 mi northwest of Fairfield in Camas County, Idaho, near the confluence of the Big Smokey and Little Smokey
creeks.

 The cabin was built in 1921-22 for John Skillern and his wife, who used it as a summer home and headquarters for John's large sheep ranching business.

 Skillern's wife based the cabin's rustic design off of the Old Faithful Inn in Yellowstone National Park, which is reflected in the cabin's steep roof and
covered front porch.

 The cabin's other significant rustic features include its sleeping lofts with pole railings, its horizontal log construction with exposed logs on the inner
walls, and its stone chimney.
</item 2>

<item 3>
Rachel, Nevada
Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.

 As of the 2010 census it had a population of 54.

 As the closest habitation to the Nellis Air Force Range and Area 51, Rachel enjoys a modest celebrity, particularly among aviation enthusiasts and UFO hunters.

 North of the town is the Quinn Canyon Range, which has the ghost town of Adaven.
</item 3>

<item 4>
Squire Creek Country Club
The Squire Creek Country Club is a private, members-only country club located in Choudrant, Louisiana, five miles northeast of Ruston.

 Squire Creek features an 18-hole championship golf course designed by Tom Fazio.

 The course has been ranked as number one in the state of Louisiana four times, and was ranked as the No. 5 best new course in the United States by "Golf
Digest".

 Squire Creek is the home golf course for the Louisiana Tech Bulldogs golf team.

 Squire Creek offers a golf training facility, which features dual bays with retractable doors for indoor or outdoor use, motion analysis, and launch monitor.

 The Squire Creek tennis facility has 6 lighted courts (4 Hydro Courts and 2 Hard Courts).

 Squire Creek has full service golf and tennis shops.

 The Squire Creek fitness facility has 2 exercise rooms and offers massage therapy.

 The Squire Creek Clubhouse features three dining areas: the more formal Main Dining Hall, the casual 19th Hole, and the Fazio Grill.

 The Squire Creek Lodge features two floors offering a combined total of seven bedrooms.
</item 4>

<item 5>
Groom Lake (salt flat)
Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).

 It is located immediately north of the Area 51 USAF installation.

 The lake at 4409 ft elevation is approximately 3.7 mi from north to south and 3 mi from east to west at its widest point, and is approximately 11.3 miles in
circumference.

 Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
</item 5>

<item 6>
Groom Range
The Groom Range is a mountain range in Lincoln County, Nevada.

 It is located within the Nevada Test and Training Range, north of Groom Dry Lake.

 The highest point in the Groom Range is 9,249 feet.

 The Groom Range is situated 26.6 miles north of the dry Groom Lake.
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
Crystal Lake (Gilmanton, New Hampshire)
Crystal Lake is a 455 acre water body located in Belknap County in the Lakes Region of central New Hampshire, United States, in the town of Gilmanton.

 Crystal Lake is at the head of the Suncook River watershed.

 Water flows into Crystal Lake from the Belknap Range through Manning and Sunset lakes to the north.

 Water from Manning Lake enters Crystal Lake via Nelson Brook.

 Other waterways flowing into Crystal Lake include Wasson Brook and Mill Brook.

 Water flows out of a dam from the southern end of Crystal Lake via the Suncook Lakes and Suncook River to the Merrimack River.

 Shoreline development along Crystal Lake consists primarily of summer cottages, with a few year-round residents.

 Belknap Mountain and Mount Major can be seen from the shoreline.
</item 8>

<item 9>
Tonopah Test Range
Tonopah Test Range (TTR), is a restricted military installation located about 30 mi southeast of Tonopah, Nevada.

 It is part of the northern fringe of the Nellis Range, measuring 625 sqmi .

 Tonopah Test Range is located about 70 mi northwest of Groom Dry Lake, the home of the Area 51 facility.

 Like the Groom Lake facility, Tonopah is a site of interest to conspiracy theorists, mostly for its use of experimental and classified aircraft.

 As such, it is not the focus of alien enthusiasts, unlike its neighbor.

 It is currently used for nuclear weapons stockpile reliability testing, research and development of fusing and firing systems, and testing nuclear weapon
delivery systems.

 The airspace comprises restricted area R-4809 of the Nevada Test and Training Range and is often used for military training.
</item 9>

<item 10>
Angustus Labyrinthus
Angustus Labyrinthus is a complex of intersecting valleys or ridges near the Martian south pole (in the Mare Australe quadrangle), located at 81.68° S and
63.25° W.

 It was nicknamed the "Inca City" by NASA scientists due to its superficial resemblance to a ruined city.

 Like other formations in the area, the name 'Angustus' derives from a name given by Eugene Antoniadi in 1930 to an albedo feature that corresponds with the
area.

 The name was approved in 2006.
</item 10>
=== END ITEM id=1eef963914613555 ===

=== ITEM id=2e253c36ef532eba ===
Gold sentences and facts:
Sentence 0: Mirpur University of Science & Technology (میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی) (MUST) was formerly a constituent college of University of Azad Jammu and Kashmir as University College of Engineering & Technology Mirpur (UCET Mirpur).
  Fact 0.0: Mirpur University of Science & Technology (MUST) was formerly a constituent college of University of Azad Jammu and Kashmir.
  Fact 0.1: The Urdu name of Mirpur University of Science & Technology is میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی.
  Fact 0.2: Mirpur University of Science & Technology was formerly named University College of Engineering & Technology Mirpur (UCET Mirpur).
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen is a university located in Debrecen, Hungary.
  Fact 1.1: The Hungarian name of the University of Debrecen is 'Debreceni Egyetem'.

Retrieved items:
<item 1>
Education in Hyderabad
Hyderabad is an important seat of learning in southern India.

 The city hosts two central universities, three deemed universities, and six state universities.

 Osmania University is one of the oldest universities in India.

 Many institutes for education like University of Hyderabad, Jawaharlal Nehru Technological University, Hyderabad, International Institute of Information
Technology, Indian Institute of Technology, Hyderabad, Tata Institute of Fundamental Research,Hyderabad, Icfai University, Birla Institute of Technology and
Science, Pilani – Hyderabad and medical colleges like Nizam's Institute of Medical Sciences are located there.

 Also based in the city are the Institute of Public Enterprise and the National Academy of Legal Studies & Research (NALSAR).

 Hyderabad has various research institutes such as the Indian Institute of Chemical Technology, Centre for Cellular and Molecular Biology.

 It is also the home of Maulana Azad National Urdu University as well as Dr. B.R. Ambedkar Open University.

 This educational infrastructure attracts students from all over India and some international students (especially from Africa and the Middle East).

 Most prolific business school in India Nmims is also expanding its roots in Hyderabad.
</item 1>

<item 2>
University of Delaware Botanic Gardens
The University of Delaware Botanic Gardens are botanical gardens and an arboretum located on the campus of the University of Delaware, in Newark, Delaware,
United States.

 The gardens are open to the public without charge.
</item 2>

<item 3>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 3>

<item 4>
Bangladesh University of Business and Technology
Bangladesh University of Business and Technology (Bengali: বাংলাদেশ ইউনিভার্সিটি অফ বিজনেস অ্যান্ড টেকনোলজি ) or BUBT is one of the eight private universities
to receive the green Signal from the government, located in Mirpur, Dhaka, Bangladesh.

 The university was established under the Private University Act 1992.

 BUBT is regulated by the Bangladesh University Grants Commission (UGC).
</item 4>

<item 5>
Mirpur University of Science and Technology
Mirpur University of Science & Technology (میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی) (MUST) was formerly a constituent college of University of Azad Jammu and
Kashmir as University College of Engineering & Technology Mirpur (UCET Mirpur).

 It is a state university and the President of Azad Jammu & Kashmir is the Chancellor of the university.

 The Vice-Chancellor is the executive head and manages the university functions.
</item 5>

<item 6>
Albanian University in Berat
Through presentations, research and exchange, the University directs and encourages the identification and development of projects that will once again
consolidate its primary purpose, scientific development, and research toward better quality.
</item 6>

<item 7>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 7>

<item 8>
Albanian University in Berat
The University of Berat is a private institution, independent and secular Higher Education, which operates the state fee.

 Licensed by the Council of Ministers, no. 697 dated 18.06.2009, he opened the doors in the academic year 2009- 2010, offering programs in two faculties:
Faculty of Social Sciences Programs in Science Teaching Legal, Political Science and Psychology General Administration and the Faculty of Economics with
Learning programs in Banking and Finance-Business Administration.

 During the first academic year, about 1,000 students attended studies.

 A year later, this new institution offering undergraduate degrees in study programs: Preschool Teacher Education, Primary Education Teacher, English Language,
Nursing, Engineering Computer and Information Technology, thus finalizing the 2010-2011 academic year with the opening of two new faculties: Faculty of Medical
Sciences and the Faculty of Architecture and Inxhinierive.

 Academic Year 2010-2011 finds the University Berat with about 2600 students in the first cycle of study.

 Since the first year of its inception, the institution pointed at the center of academic life, the promotion of intellectual product and melting learning and
research.
</item 8>

<item 9>
Commonwealth System of Higher Education
The Commonwealth System of Higher Education is a statutory designation by the Commonwealth of Pennsylvania that confers "state-related" status on four
universities located within the state.

 The designation establishes the schools as an "instrumentality of the commonwealth" and provides each university with annual, non-preferred financial
appropriations in exchange offering tuition discounts to students that are residents of Pennsylvania and a minority state-representation on each school's board
of trustees.

 Legally, however, the universities remain separate and private entities, operating under their own charters, governed by independent boards of trustees, and
with its assets under their own ownership and control thereby retaining much of the freedom and individuality of private institutions, both administratively and
academically.

 It is the only public-private hybrid system of higher education in the United States that is so construed, although Cornell University, the University of
Delaware, and Rutgers University represent alternative types of public-private university hybrids.
</item 9>

<item 10>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 10>
=== END ITEM id=2e253c36ef532eba ===

=== ITEM id=5585a09bf35eb144 ===
Gold sentences and facts:
Sentence 0: The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
  Fact 0.0: The Albanian Coalition 'Perspective' is an Albanian political party.
  Fact 0.1: The Albanian Coalition 'Perspective' is in Montenegro.
Sentence 1: Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.
  Fact 1.0: Albanians in Montenegro constitute 4.91% of the county's total population.
  Fact 1.1: The Albanian name for Albanians in Montenegro is 'Shqiptarët e Malit të Zi'.

Retrieved items:
<item 1>
Minority languages of Montenegro
European Charter for Regional or Minority Languages entered into force in Montenegro in June 2006, following the independence of Montenegro from the State Union
of Serbia and Montenegro on 3 June 2006.

 Constitution of Montenegro from 2007 states that Montenegrin language is official language of country, while Bosnian, Croatian, Serbian and Albanian are
languages in official use.

 The Constitution states that languages in official use are those of groups that form at least 1% of the population of Montenegro, as per 2003 population
census.

 The Law on National Minorities specifies that the percentage of members of national minorities in total population of the local government should be 15% in
order for their language and script to be introduced in official use.

 Media whose founder is Montenegro government are obliged to broadcast news, cultural, educational, sports and entertainment programs in minority languages.

 Minorities and their members have the right to education in their language in regular and vocational education.
</item 1>

<item 2>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 2>

<item 3>
Albanians in the Republic of Macedonia
Albanians are the largest ethnic minority in the Republic of Macedonia.

 Of the 2,022,547 citizens of Macedonia, 509,083, or 25.2%, are Albanian according to the latest national census in 2002.

 The Albanian minority lives mostly in the north-western part of the country.

 The largest Albanian communities are in the municipalities of Tetovo (70.3% of the total population), Gostivar (66.7%), Debar (58.1%), Struga (56.8%), Kičevo
(54.5%), Kumanovo (25.8%) and Skopje (20.4%).
</item 3>

<item 4>
Albania
Albania ( , ; Albanian: "Shqipëri/Shqipëria" ; Gheg Albanian: "Shqipni/Shqipnia or Shqypni/Shqypnia" ), officially the Republic of Albania (Albanian: "Republika
e Shqipërisë" , ] ), is a country in Southern and Southeastern Europe.

 The country spans 28,748 km2 and had a total population of almost 3 million people as of 2016 .

 Albania is located in the southwestern part of the Balkan Peninsula, bordered by Montenegro to the northwest, Kosovo to the northeast, the Republic of
Macedonia to the east, and Greece to the south and southeast.

 The country has a coastline on the northern shore of the Mediterranean Sea, the Adriatic Sea to the west and the Ionian Sea to the southwest, forming the
Albanian Riviera.

 Albania is less than 72 km from Italy, across the Strait of Otranto which connects the Adriatic Sea to the Ionian Sea.
</item 4>

<item 5>
Darza
Darza (Albanian: "Darzë" , Montenegrin and ) is a village in the Ulcinj Municipality, southeastern Montenegro.

 It is a multi-ethnic settlement, inhabited by Montenegrins, Serbs and Albanians.

 According to the 2003 census, the total population was 119.
</item 5>

<item 6>
Orlov let
Orlov let ("Eagle's Flight") was the name of an anti-terrorist operation conducted by the Montenegrin police to arrest a group of Albanians who planned
terrorist attacks and an armed conflict in Albanian-inhabited parts of Montenegro.

 The group of 17 people planned operations from 2004 to their arrest on the night of 10/11 September 2006.

 Weapons and explosive material were found in police raids.

 The group's main operative was the destroying of cultural and religious buildings (Orthodox) in Albanian-inhabited territory.

 The group was organised by an Albanian association based in Detroit, and members of the Kosovo Liberation Army, from which they received financial aid and
smuggled arms into Montenegro.

 The group was given a total sentence of 51 years of prison.

 Relations between the state and the Albanian minority strained after the event.
</item 6>

<item 7>
Albanians
Albanians (Albanian: "Shqiptarët" ) are a nation and ethnic group, native to Albania, Kosovo and other countries who share a common culture, ancestry and speak
the Albanian language as a native tongue.

 Legally, the term is used to refer to the citizens of the Republic of Albania.

 Ethnic Albanians speak the Albanian language and more than half of ethnic Albanians live in Albania and Kosovo.

 A large Albanian population lives in Greece, Italy, the Republic of Macedonia, with smaller Albanian populations located in Serbia and Montenegro.
</item 7>

<item 8>
Albanians in Montenegro
Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.

 Albanians of Montenegro are Ghegs who mainly live in southeastern and eastern Montenegro, mainly in the following municipalities: Ulcinj (71% of population),
Plav (19%), Bar (6%), Podgorica (5%) and Rožaje (5%).
</item 8>

<item 9>
Albania
Albania is a unitary parliamentary constitutional republic with the capital in Tirana, the country's largest city and main economic and commercial centre,
followed by Durrës.

 The country's other major cities include Vlorë, Sarandë, Shkodër, Berat, Korçë, Gjirokastër and Fier.
</item 9>

<item 10>
Albanian Coalition &quot;Perspective&quot;
The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
</item 10>
=== END ITEM id=5585a09bf35eb144 ===

=== ITEM id=427338d4bcac479c ===
Gold sentences and facts:
Sentence 0: The Australian women's national soccer team represents Australia in international women's soccer at the senior level.
  Fact 0.0: The Australian women's national soccer team represents Australia in international women's soccer.
  Fact 0.1: The Australian women's national soccer team represents Australia at the senior level.
Sentence 1:  The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.
  Fact 1.0: The Australian women's national soccer team's official nickname is the Matildas.
  Fact 1.1: The nickname Matildas comes from the Australian folk song "Waltzing Matilda".
  Fact 1.2: The Australian women's national soccer team was known as the Female Socceroos before 1995.
Sentence 2: "Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
  Fact 2.0: "Waltzing Matilda" is Australia's best-known bush ballad.
  Fact 2.1: "Waltzing Matilda" has been described as the country's "unofficial national anthem."

Retrieved items:
<item 1>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 1>

<item 2>
The entries describe sports teams, venues, parks, and communities: Seattle Sounders FC 2, a Sounders reserve team founded in 2014; Charlotte Hornets; London
Skolars; Sacramento Republic FC, 2014 USL champions seeking MLS expansion; and several baseball, racing, and pipe-band organizations. Facilities include
Boston’s TD Garden, Braves Field site, and Camden Yards complex; BYU’s Smith Fieldhouse; Campeche’s Estadio Nelson Barrera; and Sacramento’s Papa
</item 2>

<item 3>
Spencer Park, Queensland
Spencer Park (currently known as Corporate Travel Management Stadium) is a football stadium located in the suburb of Newmarket, north of Brisbane, Queensland,
Australia.

 It is home to National Premier Leagues Queensland team Brisbane City FC.

 The stadium, while officially holding 10,000, seats only 3,000 under the grandstand, the "Gino Merlo Stand".

 The stadium opened in 1963 on the site of an old Brisbane City Council waste disposal site, and underwent a major renovation in 1981 with the construction of
the Gino Merlo Stand.

 The stadium is home to La Rustica restaurant, where traditional Italian foods, including pizza, are served.
</item 3>

<item 4>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 4>

<item 5>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 5>

<item 6>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 6>

<item 7>
The collection surveys football cups, national teams, clubs and collegiate leagues, plus surfing and baseball events and the European Environment Agency.
Highlights include the 92-club English EFL Cup; Trinidad and Tobago’s First Citizens Cup; East Asian men’s and women’s championships; the FIFA Women’s World
Cup; and histories of women’s teams from Gambia, Togo and Australia. It also covers Nordic women’s championships, the 1929–30 Home Championship, Real Sociedad
and its
</item 7>

<item 8>
Australia (Gyroscope song)
"Australia" is a song by Australian rock band Gyroscope, from the album "Breed Obsession".

 It was released as the third single from their highly successful album, which peaked at number one on the Australian Albums Chart.

 The song was released to radio in May 2008, and was released as a CD single and digital download on 14 June 2008, along with two acoustic pieces.

 One of those being a live acoustic version of "Australia", the other a live acoustic cover of Chris Isaak's "Wicked Game".
</item 8>

<item 9>
Vanuatu national under-23 football team
The Vanuatu national under-23 football team, also known as Vanuatu Cyclone, represents Vanuatu at U23 tournaments.

 The team is considered to be the feeder team for the Vanuatu national football team.

 They are controlled by the Vanuatu Football Federation.

 The team has gained notoriety for thrashing Micronesia's side 46–0 in the 2015 Pacific Games.
</item 9>

<item 10>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 10>
=== END ITEM id=427338d4bcac479c ===

=== ITEM id=a0956d672d3fa2b0 ===
Gold sentences and facts:
Sentence 0: Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist Tayeb Salih.
  Fact 0.0: Season of Migration to the North is a classic post-colonial Sudanese novel.
  Fact 0.1: Season of Migration to the North is written by the novelist Tayeb Salih.
  Fact 0.2: The Arabic title of Season of Migration to the North is موسم الهجرة إلى الشمال, transliterated Mawsim al-Hiǧra ilā ash-Shamāl.
Sentence 1:  The novel is a counter narrative to "Heart of Darkness".
  Fact 1.0: Season of Migration to the North is a counter narrative to Heart of Darkness.
Sentence 2: Heart of Darkness (1899) is a novella by Polish-British novelist Joseph Conrad, about a voyage up the Congo River into the Congo Free State, in the heart of Africa, by the story's narrator Charles Marlow.
  Fact 2.0: Heart of Darkness is dated 1899.
  Fact 2.1: Heart of Darkness is a novella.
  Fact 2.2: Heart of Darkness is by Polish-British novelist Joseph Conrad.
  Fact 2.3: Heart of Darkness is about a voyage up the Congo River into the Congo Free State.
  Fact 2.4: The voyage in Heart of Darkness goes into the heart of Africa.
  Fact 2.5: The narrator of Heart of Darkness is Charles Marlow.

Retrieved items:
<item 1>
First North Americans
First North Americans are a series of historical fiction novels published by Tor and written by husband and wife co-authors W. Michael Gear & Kathleen O'Neal
Gear.

 The series, which began with 1990's "People of the Wolf", explores various civilizations and cultures in prehistoric North America.

 It is comparable to Jean M. Auel's "Earth's Children" series, set in prehistoric Europe, but each of its books focuses on a different time period, location,
and set of characters.

 The first four novels form a coherent, more or less linear narrative, from the initial migration of Siberian peoples into what is now Canada and Alaska (People
of the Wolf) through the florescence of the Mississippian semi-urban mound-building culture, considered the "high-water mark" of North American pre-Columbian
civilization, around 1000 AD.

 The remaining novels cover a wide variety of times and settings, in no particular order, ranging from tropical Florida in the 6th millennium BC to the Chaco
Empire of the Southwest in the 13th century AD.
</item 1>

<item 2>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
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
Season of Migration to the North
Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist
Tayeb Salih.

 Originally published in Arabic in 1966, it has since been translated into more than twenty languages.

 Salih was fluent in both English and Arabic, but chose to pen this novel in Arabic.

 The English translation was published in 1969 as part of the influential Heinemann African Writers Series.

 The novel is a counter narrative to "Heart of Darkness".

 It was described by Edward Said as one of the six great novels in Arabic literature.

 In 2001 it was selected by a panel of Arab writers and critics as the most important Arab novel of the twentieth century.
</item 4>

<item 5>
First North Americans
The novels take into account new developments in North American archaeology such as the discovery of Kennewick Man and the development of the coastal-route
model as a possible alternative or supplement to overland migration across Beringia.
</item 5>

<item 6>
Heart of Darkness
Heart of Darkness (1899) is a novella by Polish-British novelist Joseph Conrad, about a voyage up the Congo River into the Congo Free State, in the heart of
Africa, by the story's narrator Charles Marlow.

 Marlow tells his story to friends aboard a boat anchored on the River Thames, London, England.

 This setting provides the frame for Marlow's story of his obsession with the ivory trader Kurtz, which enables Conrad to create a parallel between London and
Africa as places of darkness.
</item 6>

<item 7>
Waiting for Happiness
Waiting for Happiness (original title: Heremakono) is a 2002 drama film written and directed by Abderrahmane Sissako.

 Main characters are a student, who has returned to his home in Nouadhibou, an electrician and his child apprentice, and the local women.

 The film is characterized by a succession of scenes of the daily life of the characters which are unique to their particular African and Arab cultures, while
borrowing from tropes of Tayeb Saleh's Season of Migration to the North (موسم الهجرة إلى الشمال).

 The viewer must interpret the scenes without much help from narrator or plot, while the structure of the film hangs on a series of mundane but visually
arresting moments, many of which are repeated in other works in Abderrahmane Sissako's opus, including scenes at a barber shop and a photo booth, also present
in his earlier La Vie Sur Terre and later Timbuktu.
</item 7>

<item 8>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 8>

<item 9>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 9>

<item 10>
Paradox (TV series)
Paradox is a 2009 British science fiction police drama, starring Tamzin Outhwaite as Detective Inspector Rebecca Flint.

 Written by Lizzie Mickery and produced by Clerkenwell Films for the BBC, it was filmed and set in Manchester, England.
</item 10>
=== END ITEM id=a0956d672d3fa2b0 ===

=== ITEM id=fa2528bc04e26f84 ===
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
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 1>

<item 2>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 2>

<item 3>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 3>

<item 4>
Ta'if Regional Airport
Ta'if Regional Airport (IATA: TIF, ICAO: OETF) is an airport in Ta'if, Saudi Arabia.

 Despite its name, it also offers many international flights as well.

 The airport is located 30 km to the east of Taif and 70 km from Mecca.

 The airport is considered important in Saudi aviation history as it witnessed the first landing of Ibn Saud's plane, founder of the Kingdom.

 It was converted to a regional airport in 2009 when GACA allowed international airlines to operate at the airport as the city's population was increasing and
to reduce pressure on the three main airports at the time.

 Despite being named as a "Regional" airport, the airport actually has international flights destinations in half a dozen countries outside of Saudi Arabia.
</item 4>

<item 5>
Brainerd Lakes Regional Airport is a city- and county-owned, general-aviation airport near Brainerd, Minnesota, with one commercial airline. Grand Sky,
established in 2015 at Grand Forks Air Force Base, is a 217-acre UAS business park with runway access. Grand Forks Municipal Airport closed after World War II
and became urban land; Grand Forks International Airport serves the area and offers customs but no scheduled international flights. Nearby Emerado had 414
residents in
</item 5>

<item 6>
Yakutat Airport
Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business
district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.

 Airline service is subsidized by the Essential Air Service program.
</item 6>

<item 7>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 7>

<item 8>
Parma’s Giuseppe Verdi Airport opened in 1991, 1.3 nautical miles from the city. San Antonio, Texas, features SAT airport, a 10-story Crowne Plaza, Trinity
University, M7 Aerospace, SyberJet facilities, the Artist Foundation (arts grants; annual ArtBall), and historic Hertzberg Clock. A separate San Antonio
International was a soccer club. South America’s San Antonio River forms part of the Argentina–Brazil border; Texas’s San
</item 8>

<item 9>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 9>

<item 10>
The entries describe U.S. and Saudi airports and authorities: Atlanta South (formerly Tara Field); remote Yakutat, served through Essential Air Service;
Wilmington, NC; Pennsylvania’s SARAA, overseeing four airports; Ta’if, a regional airport with international flights; Raleigh Exec, a reliever; Texarkana,
serving general aviation and American Eagle; Pensacola, despite its name, without direct international flights; Wilmington, Delaware; Columbus (formerly Port
Columbus); Amarillo, renamed for
</item 10>
=== END ITEM id=fa2528bc04e26f84 ===

=== ITEM id=7dbb29fb6ffcad63 ===
Gold sentences and facts:
Sentence 0: Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.
  Fact 0.0: Wenling is a coastal county-level city.
  Fact 0.1: Wenling is in the municipal region of Taizhou.
  Fact 0.2: Taizhou is in southeastern Zhejiang province.
  Fact 0.3: Wenling is in China.
  Fact 0.4: In the Wenling dialect, Wenling is called Uen-lin Zy.
Sentence 1: Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.
  Fact 1.0: Ruili is a county-level city of Dehong Prefecture.
  Fact 1.1: Ruili is in the west of Yunnan province.
  Fact 1.2: Yunnan province is in the People's Republic of China.
  Fact 1.3: Ruili's Burmese name is ရွှေလီ.
  Fact 1.4: Ruili's Thai name is เมืองมาว.

Retrieved items:
<item 1>
Ruili Border Economic Cooperation Zone
Ruili Border Economic Cooperation Zone (RLBECZ) is a Chinese State Council-approved Industrial Park based in Ruili City, Dehong Prefecture, Yunnan, China,
founded in 1992 and was established to promote trade between China and Myanmar.

 The zone connects Ruili with Man-wing City.
</item 1>

<item 2>
G56 Hangzhou–Ruili Expressway
The Hangzhou–Ruili Expressway (), commonly referred to as the "Hangrui Expressway" () is an expressway in China that connects the cities of Hangzhou, Zhejiang,
and Ruili, Yunnan, a city on the border with Burma.

 When complete, it will be 2935 km in length.
</item 2>

<item 3>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 3>

<item 4>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 4>

<item 5>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 5>

<item 6>
Linzhou, Henan
Linzhou (), formerly Lin County (林县), is a county-level city in Anyang, Henan, China.

 Adjacent to Shanxi Province and Hebei Province, it is located in the north of Henan Province and at the eastern foot of the Taihang Mountains.

 It covers an area of 2046 square kilometers and has a population of about one million.
</item 6>

<item 7>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 7>

<item 8>
Dali–Ruili Railway
The Dali–Ruili Railway or Darui Railway (), is a single-track electrified railroad under construction in Yunnan Province of Southwest China.

 The line is slated to run 336.39 km from Dali in central Yunnan to Ruili in southwestern Yunnan on the border with Myanmar.

 The line traverses rugged terrain, and bridges and tunnels will account for 75% of the total track length, including a 36-km tunnel through the Gaoligong
Mountains.

 Construction began in May 2011 and is scheduled to take six years.

 Cities and towns along route include Dali, Yangbi Yi Autonomous County, Yongping County, Baoshan, Mang City (also known as Mangshi or Dehong) and Ruili.
</item 8>

<item 9>
Ruili
Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.

 It is a major border crossing between China and Myanmar, with the town of Muse located across the border.
</item 9>

<item 10>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 10>
=== END ITEM id=7dbb29fb6ffcad63 ===

=== ITEM id=9b7b4dbeb28e8733 ===
Gold sentences and facts:
Sentence 0: Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in China and Japan.
  Fact 0.0: Pterostyrax is also called the epaulette tree.
  Fact 0.1: Pterostyrax is a small genus.
  Fact 0.2: Pterostyrax has four species.
  Fact 0.3: Pterostyrax species are deciduous.
  Fact 0.4: Pterostyrax species are large shrubs or small trees.
  Fact 0.5: Pterostyrax is in the family Styracaceae.
  Fact 0.6: Pterostyrax is native to eastern Asia.
  Fact 0.7: Pterostyrax is native to China.
  Fact 0.8: Pterostyrax is native to Japan.
Sentence 1: Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.
  Fact 1.0: Dregea is a genus of vines.
  Fact 1.1: Dregea is in the family Apocynaceae.
  Fact 1.2: Dregea was first described as a genus with this name in 1838.
Sentence 2:  It is native to Africa and southern Asia.
  Fact 2.0: Dregea is native to Africa.
  Fact 2.1: Dregea is native to southern Asia.

Retrieved items:
<item 1>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 1>

<item 2>
Rhabdodontidae
Rhabdodontids were herbivorous ornithopod dinosaurs from the Cretaceous Period.

 Rhabdodontids were similar to large, robust "hypsilophodonts", with deep skulls and jaws.

 The family was first proposed by David B. Weishampel and colleagues in 2002.

 The rhabdodontids were defined as "the most recent common ancestor of "Zalmoxes robustus" and "Rhabdodon priscus" and all the descendants of this common
ancestor".

 In 2005, Paul Sereno defined the family as "the most inclusive clade containing "Rhabdodon priscus" but not "Parasaurolophus walkeri"".

 The Rhabdodontidae includes the type genus "Rhabdodon", "Zalmoxes", "Mochlodon" and possibly "Muttaburrasaurus".

 Rhabdodontid fossils have been found in Europe and Australia in formations dating from the Early to Late Cretaceous, 100 to 66 million years ago.

 Ősi "et al."
</item 2>

<item 3>
Dregea
Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.

 It is native to Africa and southern Asia.
</item 3>

<item 4>
Dregea sinensis
Dregea sinensis is a species of plant in the Apocynaceae family that is native to China.
</item 4>

<item 5>
Pterostyrax hispidus
Pterostyrax hispidus, the epaulette tree or fragrant epaulette tree, is a species of flowering plant in the family Styracaceae, native to China and Japan.

 Growing to 15 m tall by 12 m broad, it is a substantial, spreading, deciduous shrub with oval leaves up to 20 cm long, and clusters of pure white, fragrant,
pendent flowers in summer.
</item 5>

<item 6>
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 6>

<item 7>
Rhabdodontidae
Ősi "et al." (2012) proposed that "Rhabdodon" underwent gigantism on the mainland, as opposed to "Zalmoxes" and "Mochlodon" experiencing nanism on island
habitats.

 In 2013, Darren Naish discovered a tiny rhabdodont tibia from Sebeș, Romania, turning out to be the smallest known; a paper (Brusette et "al" 2013.)

 later in the same year identified it as "Zalmoxes", cf. "Z. shqiperorum".
</item 7>

<item 8>
Dregea volubilis
Dregea volubilis is a species of plant in the Apocynaceae family that is native to India.
</item 8>

<item 9>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 9>

<item 10>
Styrax crotonoides
Styrax crotonoides is a species of plant in the genus "Styrax" and family Styracaceae.

 It is found in Peninsular Malaysia and Singapore.
</item 10>
=== END ITEM id=9b7b4dbeb28e8733 ===

=== ITEM id=9645180125c58c43 ===
Gold sentences and facts:
Sentence 0: "Love Incredible" is a song by Norwegian DJ and record producer Cashmere Cat, featuring vocals by Cuban-American singer Camila Cabello.
  Fact 0.0: 'Love Incredible' is a song by Norwegian DJ and record producer Cashmere Cat.
  Fact 0.1: 'Love Incredible' features vocals by Cuban-American singer Camila Cabello.
Sentence 1:  It was released on 17 February 2017 through Interscope and Mad Love as the third single from his debut album, "9".
  Fact 1.0: 'Love Incredible' was released on 17 February 2017.
  Fact 1.1: 'Love Incredible' was released through Interscope and Mad Love.
  Fact 1.2: 'Love Incredible' was the third single from the debut album '9' of Cashmere Cat.
Sentence 2:  An imprint of Interscope Geffen A&M Records, its parent company is Universal Music Group, a subsidiary of Vivendi S.A.
  Fact 2.0: An imprint of Interscope Geffen A&M Records has Universal Music Group as its parent company.
  Fact 2.1: Universal Music Group is a subsidiary of Vivendi S.A.

Retrieved items:
<item 1>
Kiss (Carly Rae Jepsen album)
Kiss is the second studio album by Canadian recording artist Carly Rae Jepsen.

 It was released on September 14, 2012, by 604, Schoolboy and Interscope Records.

 After her debut, "Tug of War" (2008), which managed to receive success in Canada but was never released worldwide, "Kiss" became Jepsen's first internationally
released album.

 Songs on the album are in the nu-disco, dance-pop, and teen pop genres, drawing inspiration from The Cars, Madonna, and Swedish performer, Robyn.

 Featuring production from a wide collection of producers including Dallas Austin, Josh Ramsay, and Redfoo, the album features a track with guest vocals from
Justin Bieber.
</item 1>

<item 2>
Kinetic Records
The label was distributed by Reprise/Warner Bros.

 Records from its inception until 2001, when Kinetic switched distribution to BMG.

 The latter eventually absorbed the label during the merger with Sony Music.
</item 2>

<item 3>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 3>

<item 4>
Dirty Dancer
"Dirty Dancer" is a song by Spanish recording artist Enrique Iglesias and American recording artist Usher, recorded for Iglesias' sixth studio album,
"Euphoria", and included on Usher's EP "Versus".

 A remix of the song, which features American rapper Lil Wayne and American singer Nayer, was released on 9 May 2011 by Universal Music Group as the sixth
single from "Euphoria".

 Written by the two singers with Evan Bogart, Erika Nuri and David Quiñones and producer RedOne, "Dirty Dancer" is a dance-pop song with rave music influences.
</item 4>

<item 5>
Stay the Night (Zedd song)
"Stay the Night" is a song by Russian-German electronic dance music producer Zedd, from the deluxe edition (2013) of his debut studio album, "Clarity " (2012).

 It features vocals from Hayley Williams, the lead singer of American rock group Paramore.

 The song was written by Zedd, Williams, Benjamin Eli Hanna, and Carah Faye.

 "Stay the Night" was released to digital retailers on September 10, 2013, by Interscope Records as the lead single off the deluxe edition of "Clarity", and the
fourth overall single from the album.
</item 5>

<item 6>
Hoodlum (soundtrack)
Hoodlum is the soundtrack to the 1997 American crime drama film "Hoodlum".

 It was released on August 12, 1997 by Interscope Records and consisted of a blend of hip hop and R&B music.

 The soundtrack peaked at 94 on the "Billboard" 200 and 23 on the Top R&B Albums and contained Mobb Deep's single "Hoodlum" which went to 29 on the Hot Rap
Singles.
</item 6>

<item 7>
Interscope Records
Interscope Records is an American major record label.

 An imprint of Interscope Geffen A&M Records, its parent company is Universal Music Group, a subsidiary of Vivendi S.A.
</item 7>

<item 8>
Kinetic Records
Kinetic Records was a New York City-based record label founded by Steve Lau (founding member of Sire Records act the Ocean Blue), that launched the careers of
artists and DJs such as Paul Oakenfold, BT, Ray Munns, Deepsky, Kosheen, Tilt, Timo Maas, Grace, Sandra Collins, Billy Thermal, Binary Finary, and Shpongle.

 Other successful artists on the label included Ash, Sasha and John Digweed, South, Faithless, DJ DAN, LTJ Bukem, Dario G, Luomo, Infected Mushroom, Max Graham,
Dave Ralph Music Instructor, and Hybrid.

 It was also well known for its Tranceport series, of which Paul Oakenfold's first installment is considered to be the breakthrough record for the trance genre
with the highest sales of any mix compilation.

 The release of Sasha's Airdrawndagger and Sasha and Digweed's mix compilation "Communicate" were also big releases for the label.

 Kinetic was the exclusive distributor of the Perfecto label in the US as well as the Another Late Night compilation series, which featured Howie B., Fila
Brazillia, and Rae and Christian.
</item 8>

<item 9>
Love Incredible
"Love Incredible" is a song by Norwegian DJ and record producer Cashmere Cat, featuring vocals by Cuban-American singer Camila Cabello.

 It was released on 17 February 2017 through Interscope and Mad Love as the third single from his debut album, "9".
</item 9>

<item 10>
Encore (Eminem album)
Encore (stylized as ƎNCORE) is the fifth studio album by American rapper Eminem.

 It was released by Aftermath Entertainment, Shady Records, and Interscope Records.

 Its release was set for November 16, 2004, but was moved up to November 12 (coincidentally, exactly eight years to the day since his debut album, "Infinite",
was released) after the album was leaked to the Internet.

 "Encore" sold 710,000 copies in its first three days, and went on to sell over 1.5 million copies in its first two weeks of release in the United States,
certified quadruple-platinum that mid-December.

 Nine months after its release, worldwide sales of the album stood at 11 million copies.

 By December 2016, the album had sold over 5 million copies in the United States and more than 23 million copies worldwide.
</item 10>
=== END ITEM id=9645180125c58c43 ===

=== ITEM id=69e765ad2da1b479 ===
Gold sentences and facts:
Sentence 0:  He appeared in 57 films between 1919 and 1932.
  Fact 0.0: The person described appeared in 57 films.
  Fact 0.1: The person described appeared in 57 films between 1919 and 1932.
Sentence 1:  He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
  Fact 1.0: The person described directed 54 films.
  Fact 1.1: The person described directed these films between 1920 and 1933.
  Fact 1.2: The person's directed films include one Harry Langdon short.
  Fact 1.3: The person's directed films include two of the Tay Garnett-penned comedies Stan Laurel made for Joe Rock.
  Fact 1.4: The person's directed films include fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
Sentence 2: John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.
  Fact 2.0: John Scott Biddle lived from 1925 to 2008.
  Fact 2.1: John Scott Biddle was a foremost yachting cinematographer.
  Fact 2.2: John Scott Biddle was a lecturer.
  Fact 2.3: John Scott Biddle established a film-making career.
  Fact 2.4: John Scott Biddle's film-making career spanned more than forty years.

Retrieved items:
<item 1>
John Biddle (Unitarian)
John Biddle or Bidle (born Wotton-under-Edge, Gloucestershire, England, 14 January 1615 – died 22 September 1662) was an influential English nontrinitarian, and
Unitarian.

 He is often called "the Father of English Unitarianism".
</item 1>

<item 2>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 2>

<item 3>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 3>

<item 4>
Charles John Biddle
Charles John Biddle (April 30, 1819 – September 28, 1873) was an American soldier, lawyer, congressman, and newspaper editor.
</item 4>

<item 5>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 5>

<item 6>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 6>

<item 7>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 7>

<item 8>
Biddle family
The American Biddle family of Philadelphia began when William Biddle (1630–1712) and Sarah Kempe (1634–1709), Quakers, emigrated from England to America in
1681, in part, to avoid religious persecution.

 Having acquired extensive rights to over 43000 acre of lands in Quaker West Jersey, they settled first at Burlington.

 Two third generation brothers, William Biddle, 3rd (1698–1756), and John Biddle (1707–1789), moved from Mount Hope (1684) near Bordentown to Philadelphia in
the 1720s and 1730s and constituted the first generation of the Philadelphia Biddle family.
</item 8>

<item 9>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 9>

<item 10>
John Biddle (Michigan politician)
John Biddle (March 2, 1792 – August 25, 1859) was a delegate to the United States Congress from the Michigan Territory.
</item 10>
=== END ITEM id=69e765ad2da1b479 ===
