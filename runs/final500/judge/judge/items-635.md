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

=== ITEM id=b68ca2538cb2ba7c ===
Gold sentences and facts:
Sentence 0: St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.
  Fact 0.0: St. Elizabeths Hospital opened in 1855.
  Fact 0.1: St. Elizabeths Hospital opened as the first federally operated psychiatric hospital in the United States.
Sentence 1:  Founded in 1861, it is the longest continuously operating hospital in the District.
  Fact 1.0: St. Elizabeths Hospital was founded in 1861.
  Fact 1.1: St. Elizabeths Hospital is the longest continuously operating hospital in the District.

Retrieved items:
<item 1>
Westminster St Margaret and St John
St Margaret was an ancient parish in the City and Liberty of Westminster and the county of Middlesex.

 It included the core of modern Westminster, including the Palace of Westminster and the area around, but not including Westminster Abbey.

 In 1727 it was divided into St Margaret's and St John's, to coincide with the building of the Church of St John the Evangelist, constructed by the Commission
for Building Fifty New Churches in Smith Square to meet the demands of the growing population, but there continued to be a single vestry for the parishes of St
Margaret and St John.

 This was reformed in 1855 by the Metropolis Management Act, and the two parishes formed the Westminster District until 1887.

 In 1889 St Margaret and St John became part of the County of London.

 The vestry was abolished in 1900, to be replaced by Westminster City Council, but St Margaret and St John continued to have a nominal existence until 1922.
</item 1>

<item 2>
Vassar Brothers Medical Center
Vassar Brother Medical Center (locally known as Vassar Hospital or VBMC) is a major medical facility located in the city of Poughkeepsie, New York that is a
member of the Health Quest network, a nonprofit family of hospitals and healthcare centers in the Hudson Valley area.

 VBMC is one of two major medical centers located in Dutchess County, New York, the other being Mid-Hudson Regional Hospital of Westchester Medical Center.

 The hospital was first incorporated in 1882 under the name Vassar Brothers Hospital, following contributions made by John Guy Vassar following the death of his
brother.

 The hospital opened its doors in 1887, and initially had 40 beds divided up into four wards, each containing ten beds.

 At the time the hospital opened it also contained a labor and delivery ward, a children's ward, a nursery, three private rooms, and two isolation rooms.

 Several major renovations occurred in 1983, 1991, and 2001; adding critical care areas, a new operating and delivery wing, and a cancer care center,
respectively.
</item 2>

<item 3>
History of psychiatric hospitals in Pittsburgh
Pittsburgh, Pennsylvania, United States, is home to a plethora of hospitals, be it for the physically ill or, what was defined as the insane and feeble minded.

 In 1891, the St. Margaret Memorial Hospital opened under John H. Shoenberger.

 In 1908, the Pittsburgh Hospital was run under C. A. Fagan, and on June 1, 1889, the South Side Hospital opened under Anna M. Rindlaub.

 Pittsburgh, commonly known as the "Steel City," also built up a reputation for its abundance of hospitals that had opened up within a fairly close time period
of each other.

 During this time, many new mental hospitals were established due to overcrowding.

 In 1899, a building to house the mentally insane, Mayview State Hospital, was constructed; but it was not long before the complex had a reputation “as a place
of sorrow” due to the lack of proper health care patients received during their stay.
</item 3>

<item 4>
Providence Hospital (Washington, D.C.)
Providence Hospital is a 408 bed hospital located in the District of Columbia.

 Founded in 1861, it is the longest continuously operating hospital in the District.

 Providence Hospital is a member of Ascension Health, the largest non-profit health care organization in the United States.
</item 4>

<item 5>
Heritage Hospital
Heritage Hospital is a hospital located in Tarboro, North Carolina.

 It is a part of the University Health Systems of Eastern Carolina (UHSEC).

 Edgecombe General Hospital opened as a county-owned hospital in 1916.

 It succeeded Pittman Hospital, which opened in 1901.

 In 1959, the Hill-Burton Act helped combine Edgecombe General Hospital, with three other facilities.

 Edgecombe County sold the hospital to Hospital Corporation of America (HCA) in 1982.

 HCA opened a 127-bed facility in 1985, named Heritage Hospital.

 UHSEC bought Heritage Hospital in 1998 from HCA.

 The hospitals focus is as a community hospital.

 The hospital has 101 general and 16 rehabilitation hospital beds.

 It has five Shared Inpatient/Ambulatory Surgery, two Endoscopy, and one C-Section operating rooms.
</item 5>

<item 6>
St. Elizabeths Hospital
St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.

 Housing over 8,000 patients at its peak in the 1950s, the hospital at one point had a fully functioning medical-surgical unit, a school of nursing, and
accredited internships and psychiatric residencies.

 Its campus was designated a National Historic Landmark in 1990.
</item 6>

<item 7>
St. John's Regional Medical Center (California)
St. John's Regional Medical Center is a hospital located in Oxnard, California in the United States, and is operated by Dignity Health, along with its sister
hospital, St. John's Pleasant Valley Hospital in Camarillo, California.

 The hospital was founded in 1912.

 St. John's Regional Medical Center and St. John's Pleasant Valley Hospital offer comprehensive medical services, including 24-hour emergency medical and
surgical services and care, cancer and oncology care and support, cardiovascular care, community outreach and screenings, diagnostic imaging services,
laboratory services, maternity and women's services, neonatal intensive care, palliative care, patient and family education, rehabilitation services, spine and
orthopedic care, weight loss surgery, wound healing and oxygen therapy, and more.

 Together, St. John's Regional Medical Center and St. John's Pleasant Valley Hospital represent the largest acute-care health organization in Ventura County.

 St. John's hospitals serve all of Ventura County and beyond, including the cities of Camarillo, Moorpark, Oxnard, Port Hueneme, Ventura, and Somis.
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
Vassar Brothers Medical Center
In 2002 Vassar Brothers Hospital officially became Vassar Brothers Medical Center.

 In September 2016, VBMC announced a new $500 million expansion which would almost double the number of available rooms.

 Construction on the expansion is expected to be completed in 2019.
</item 9>

<item 10>
Douglass (Washington, D.C.)
Douglass is a residential neighborhood in Southeast Washington, D.C., on the eastern side of St. Elizabeths Hospital, on the border of the Congress Heights
Metro Station.

 It is bounded by Suitland Parkway to the north and east, Alabama Avenue to the south, and the St. Elizabeths campus to the west.
</item 10>
=== END ITEM id=b68ca2538cb2ba7c ===

=== ITEM id=e0803d7f637440b7 ===
Gold sentences and facts:
Sentence 0: Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
  Fact 0.0: Leitneria floridana is also called corkwood.
  Fact 0.1: Leitneria floridana is the sole species in the genus Leitneria.
  Fact 0.2: Leitneria floridana is a deciduous dioecious shrub or small tree.
  Fact 0.3: Leitneria floridana is found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
Sentence 1: Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.
  Fact 1.0: Tricyrtis is a genus of Asian flowering plants.
  Fact 1.1: Tricyrtis is in the lily family.
  Fact 1.2: Tricyrtis has approximately 20 known species.

Retrieved items:
<item 1>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 1>

<item 2>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 2>

<item 3>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 3>

<item 4>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 4>

<item 5>
Traubiinae
Traubiinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 5>

<item 6>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 6>

<item 7>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 7>

<item 8>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 8>

<item 9>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 9>

<item 10>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 10>
=== END ITEM id=e0803d7f637440b7 ===

=== ITEM id=c8ae7f1214d2ad6b ===
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
HSBC Building (Hong Kong)
HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC
Holdings.

 It is located along the southern side of Statue Square near the location of the old City Hall, Hong Kong (built in 1869, demolished in 1933).

 The previous HSBC building was built in 1935 and pulled down to make way for the current building.

 The address remains as 1 Queen's Road Central (the north facing side of the building was served by Des Voeux Road, which was seashore, making Queen's Road was
the main entrance, in contrast to modern day that usually accessed the bank from Des Voeux Road).

 The building can be reached from Exit K of Central MTR Station and facing Statue Square.
</item 1>

<item 2>
HSBC Building, the Bund
The HSBC Building (also known as the Municipal Government Building) is a six-floor neo-classical building in the Bund area of Shanghai, China.

 It served as the headquarters of the Shanghai branch of The Hongkong and Shanghai Banking Corporation from 1923 to 1955, and currently houses the Shanghai
Pudong Development Bank.

 The building is situated at number 12, the Bund.

 Designed by the British architecture firm Palmer & Turner Architects and Surveyors, construction of the building lasted from 5 May 1921 to 23 June 1923.
</item 2>

<item 3>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 3>

<item 4>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 4>

<item 5>
HSBC
HSBC Holdings PLC is a British multinational banking and financial services holding company, tracing its origin to a hong in Hong Kong.

 It is the world's seventh largest bank by total assets and the largest in Europe with total assets of US$2.374 trillion (as of December 2016).

 It was established in its present form in London in 1991 by The Hongkong and Shanghai Banking Corporation Limited to act as a new group holding company.

 The origins of the bank lie mainly in Hong Kong and to a lesser extent in Shanghai, where branches were first opened in 1865.

 The HSBC name is derived from the initials of the Hongkong and Shanghai Banking Corporation.

 The company was first formally incorporated in 1866.

 The company continues to see both the United Kingdom and Hong Kong as its "home markets".
</item 5>

<item 6>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 6>

<item 7>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 7>

<item 8>
Bank Street (Hong Kong)
Bank Street () is a short street in Central, Hong Kong.

 It links Des Voeux Road Central to Queen's Road Central.

 It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong)
respectively.
</item 8>

<item 9>
Bank of Canton
Bank of Canton () was established in 1912 in Canton but registered as a British company in Hong Kong.

 This made it the first Chinese-owned bank in Hong Kong.

 At the time, all the other banks in Hong Kong were foreign, primarily British, including the locally incorporated but British-run Hongkong and Shanghai Banking
Corporation.

 Security Pacific National Bank (SPNB) bought a majority share in Bank of Canton in 1971.

 In 1988, Security Pacific succeeded in wholly acquiriing Bank of Canton, which became Security Pacific Asia Bank.

 When Bank of America acquired Security Pacific Corporation in 1993 it changed Security Pacific Asia Bank’s name to Bank of America (Asia).

 China Construction Bank acquired Bank of America (Asia) in 2006.
</item 9>

<item 10>
HSBC lions
Various headquarters and branch buildings of The Hongkong and Shanghai Banking Corporation and the HSBC Group, into which the Hongkong and Shanghai Banking
Corporation has evolved, feature a pair of lion sculptures.

 The HSBC lions have become distinctive landmarks in their own right in Hong Kong and Shanghai respectively, with a further pair to be found in London.
</item 10>
=== END ITEM id=c8ae7f1214d2ad6b ===

=== ITEM id=2d732ced9ae0fafe ===
Gold sentences and facts:
Sentence 0: Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television host/personality who "TV Guide" called "The Charlie Chaplin of Television".
  Fact 0.0: Alan Young was born Angus Young.
  Fact 0.1: Alan Young was born on November 19, 1919.
  Fact 0.2: Alan Young died on May 19, 2016.
  Fact 0.3: Alan Young was a British-born Canadian-American actor.
  Fact 0.4: Alan Young was a voice actor.
  Fact 0.5: Alan Young was a comedian.
  Fact 0.6: Alan Young was a radio and television host/personality.
  Fact 0.7: TV Guide called Alan Young "The Charlie Chaplin of Television".
Sentence 1:  Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983) and in various other films, TV series and video games until his death.
  Fact 1.0: Alan Young was the voice of Disney's Scrooge McDuck for over thirty years.
  Fact 1.1: Alan Young first voiced Scrooge McDuck in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983).
  Fact 1.2: Alan Young voiced Scrooge McDuck in various other films, TV series and video games.
  Fact 1.3: Alan Young voiced Scrooge McDuck until his death.
Sentence 2:  During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy Awards in 1951.
  Fact 2.0: During the 1940s and 1950s, Alan Young starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television.
  Fact 2.1: The television version of "The Alan Young Show" gained Alan Young two Emmy Awards in 1951.
Sentence 3:  The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.
  Fact 3.0: The cartoon is an adaptation of Charles Dickens' "A Christmas Carol".
  Fact 3.1: The cartoon stars Scrooge McDuck as Ebenezer Scrooge.

Retrieved items:
<item 1>
Scrooge (1935 film)
Scrooge is a 1935 British fantasy film directed by Henry Edwards and starring Seymour Hicks, Donald Calthrop and Robert Cochran.

 Hicks appears as Ebenezer Scrooge, the miser who hates Christmas.

 It was the first sound version of the Charles Dickens classic "A Christmas Carol", not counting a 1928 short subject that now appears to be lost.

 Hicks had previously played the role of Scrooge on the stage many times beginning in 1901, and again in a 1913 British silent film version.
</item 1>

<item 2>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 2>

<item 3>
Charles Dickens’s 1843 novella *A Christmas Carol* follows miser Ebenezer Scrooge’s transformation after visits from Marley’s ghost and the three Christmas
spirits. Its adaptations include the earliest known film (1901), silent (1913), sound (1935), Alastair Sim’s 1951 film, Disney’s 1983 featurette with Scrooge
McDuck, Jack Palance’s 1998 Western-style TV version, and Robert Zemeckis’s
</item 3>

<item 4>
A Christmas Carol (2009 film)
A Christmas Carol is a 2009 American 3D computer animated motion-capture fantasy film written and directed by Robert Zemeckis.

 It is an adaptation of the Charles Dickens story of the same name and stars Jim Carrey in a multitude of roles, including Ebenezer Scrooge as a young,
middle-aged, and old man, and the three ghosts who haunt Scrooge.

 The film also features supporting roles done by Gary Oldman, Colin Firth, Bob Hoskins, Robin Wright, and Cary Elwes.
</item 4>

<item 5>
Scrooge, or, Marley's Ghost
Scrooge, or, Marley's Ghost is a 1901 British short silent drama film, directed by Walter R. Booth, featuring the miserly Ebenezer Scrooge confronted by
Marley's ghost and given visions of Christmas past, present, and future, is the earliest known film adaptation of Charles Dickens's 1843 novel "A Christmas
Carol".

 The film, "although somewhat flat and stage-bound to modern eyes," according to Ewan Davidson of BFI Screenonline, "was an ambitious undertaking at the time,"
as, "not only did it attempt to tell an 80 page story in five minutes, but it featured impressive trick effects, superimposing Marley's face over the door
knocker and the scenes from his youth over a black curtain in Scrooge's bedroom."
</item 5>

<item 6>
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
</item 6>

<item 7>
Scar (Disney)
Scar is a fictional character who appears in Walt Disney Pictures' 32nd animated feature film "The Lion King" (1994).

 The character is voiced by English actor Jeremy Irons, while his singing voice is provided by both Irons and American actor Jim Cummings, the latter of whom
was hired to replace Irons when the former damaged his singing voice.

 Subsequently, Scar makes minor appearances in the film's sequel " "(1998) and "The Lion King 1½ "(2004), in both of which he is voiced entirely by Cummings, as
well as appearing in the Broadway musical adaptation of the film, in which the role of Scar was originated by American actor John Vickery.
</item 7>

<item 8>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 8>

<item 9>
Scrooge (1951 film)
Scrooge is a 1951 British fantasy drama film and an adaptation of Charles Dickens' "A Christmas Carol" (1843).

 It stars Alastair Sim as Ebenezer Scrooge, and was produced and directed by Brian Desmond Hurst, with a screenplay by Noel Langley.

 It was released as A Christmas Carol in the United States.
</item 9>

<item 10>
Scrooge (1913 film)
Scrooge is a 1913 British black and white silent film based on the 1843 novel "A Christmas Carol" by Charles Dickens.

 It starred Seymour Hicks as Ebenezer Scrooge.

 In the United States it was released in 1926 as "Old Scrooge".
</item 10>
=== END ITEM id=2d732ced9ae0fafe ===

=== ITEM id=b878e2243f1e137b ===
Gold sentences and facts:
Sentence 0:  Her only novel, "The Bone People", won the Booker Prize in 1985.
  Fact 0.0: Her only novel is 'The Bone People'.
  Fact 0.1: 'The Bone People' won the Booker Prize.
  Fact 0.2: 'The Bone People' won the Booker Prize in 1985.
Sentence 1:  His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science fiction's top U.S. prizes in consecutive years.
  Fact 1.0: Card's novel 'Ender's Game' was published in 1985.
  Fact 1.1: 'Speaker for the Dead' is the sequel to 'Ender's Game'.
  Fact 1.2: 'Speaker for the Dead' was published in 1986.
  Fact 1.3: 'Ender's Game' won the Hugo Award.
  Fact 1.4: 'Ender's Game' won the Nebula Award.
  Fact 1.5: 'Speaker for the Dead' won the Hugo Award.
  Fact 1.6: 'Speaker for the Dead' won the Nebula Award.
  Fact 1.7: Card is the only author to win both science fiction's top U.S. prizes in consecutive years.

Retrieved items:
<item 1>
Ender Wiggin
Andrew "Ender" Wiggin is a fictional character from Orson Scott Card's 1985 science fiction novel "Ender's Game" and its sequels ("Speaker for the Dead",
"Xenocide", "Children of the Mind", "Ender in Exile"), as well as in the first part of the spin-off series, "Ender's Shadow".

 The book series itself is an expansion, with some changes to detail, of Card's 1977 short story "Ender's Game."
</item 1>

<item 2>
Ender in Exile
Ender in Exile is a science fiction novel by Orson Scott Card, part of the "Ender's Game" series, published on November 11, 2008.

 It takes place between the two award-winning novels: "Ender's Game" and "Speaker for the Dead".

 It could also be considered a parallel novel to the first three sequels in the Shadow Saga, since the entirety of this "trilogy" takes place in the span of
"Ender in Exile".

 The novel concludes a dangling story line of the Shadow Saga, while it makes several references to events that take place during the Shadow Saga.

 From yet another perspective, the novel expands (or "replaces") the last chapter of the original novel "Ender's Game".

 On the one hand, it fills the gap right before the last chapter, and on the other hand, it fills the gap between the last chapter and the original (first)
sequel (both named "Speaker for the Dead").
</item 2>

<item 3>
Ender in Exile
"Ender in Exile" begins one year after Ender has won the bugger war, and begins with the short story "Ender's Homecoming" from Card's webzine Intergalactic
Medicine Show.

 Other short stories that were published elsewhere are included as chapters of the novel.
</item 3>

<item 4>
The Bone People
The Bone People (styled by the writer and in some editions as the bone people) is a Booker Prize-winning 1984 novel by New Zealand writer Keri Hulme.
</item 4>

<item 5>
Graph query row:
work: Ender's Game
publication_year: 1985
relationship: CREATED_BY
person: Orson Scott Card
</item 5>

<item 6>
A War of Gifts: An Ender Story
A War of Gifts: An Ender Story (2007) is a science fiction novel by Orson Scott Card.

 This book is set in Card's Ender's Game series and takes place during Ender Wiggin's time at Battle School as described in Card's novels "Ender's Game" and
"Ender's Shadow".
</item 6>

<item 7>
Shadow of the Hegemon
Shadow of the Hegemon (2001) is the second novel in the "Ender's Shadow" series (often called the Bean Quartet) by Orson Scott Card.

 It is also the sixth novel in the "Ender's Game" series.

 It is told mostly from the point of view of Bean, a largely peripheral character in the original novel "Ender's Game" but the central protagonist of the
parallel narrative "Ender's Shadow."

 "Shadow of the Hegemon" was nominated for a Locus Award in 2002.
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
Graph query row:
work: The Bone People
publication_year: 1984
relationship: CREATED_BY
person: Keri Hulme
</item 9>

<item 10>
Ender's Game (short story)
"Ender's Game" is a story by Orson Scott Card.

 It first appeared in the August 1977 issue of "Analog" magazine and was later expanded into the novel "Ender's Game".

 Although the foundation of the Ender's Game series, the short story is not properly part of the "Ender's Game" universe, as there are many discrepancies in
continuity.
</item 10>
=== END ITEM id=b878e2243f1e137b ===

=== ITEM id=f3f6b2c71784c57f ===
Gold sentences and facts:
Sentence 0: Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.
  Fact 0.0: Like many of Marivaux's plays, La Surprise de l'amour makes use of stock characters from the Commedia dell'arte.
  Fact 0.1: Many of Marivaux's plays make use of stock characters from the Commedia dell'arte.
Sentence 1:  In this play, Arlequin and Columbine are featured.
  Fact 1.0: In La Surprise de l'amour, Arlequin is featured.
  Fact 1.1: In La Surprise de l'amour, Columbine is featured.
Sentence 2: Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the Italian "Commedia dell'arte".
  Fact 2.0: Harlequin is the best-known of the zanni, the comic servant characters from the Italian Commedia dell'arte.
  Fact 2.1: Harlequin is called Arlecchino in Italian.
  Fact 2.2: Harlequin is called Arlequin in French.
  Fact 2.3: Harlequin is called Harlequin in Old French.

Retrieved items:
<item 1>
Columbina
Columbina (in Italian Colombina, meaning "little dove"; in French and English Colombine) is a stock character in the "Commedia dell'Arte".

 She is Harlequin's mistress, a comic servant playing the tricky slave type, and wife of Pierrot.

 Rudlin and Crick use the Italian spelling Colombina in "Commedia dell'arte: A Handbook for Troupes".
</item 1>

<item 2>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 2>

<item 3>
Pierrot lunaire (book)
Pierrot lunaire: rondels bergamasques ("Moonstruck Pierrot: bergamask rondels") is a collection of fifty poems published in 1884 by the Belgian poet Albert
Giraud (born Emile Albert Kayenburgh), who is usually associated with the Symbolist Movement.

 The protagonist of the cycle is Pierrot, the comic servant of the French Commedia dell'Arte and, later, of Parisian boulevard pantomime.

 The early 19th-century Romantics, Théophile Gautier most notably, had been drawn to the figure by his Chaplinesque pluckiness and pathos, and by the end of the
century, especially in the hands of the Symbolists and Decadents, Pierrot had evolved into an alter-ego of the artist, particularly of the so-called poète
maudit.

 He became the subject of numerous compositions, theatrical, literary, musical, and graphic.
</item 3>

<item 4>
Niccolò Barbieri
Niccolò (or Nicolò) Barbieri (Vercelli, 1586 - 1641) was an Italian writer and actor of the commedia dell'arte theatrical genre.

 He was also known as Beltrame di Milano ("Beltrame of Milan") in reference to one of his most popular characters, Beltrame; this was the main character of one
of Barbieri's best known plays, "L'inavertito", which is known to have inspired Moliere's "L'Étourdi ou Les Contretemps" ("The Blunderer").

 Besides popularizing Beltrame, Barbieri is also credited with creating another "commedia dell'arte" mask, Scapino.
</item 4>

<item 5>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 5>

<item 6>
Gilles (stock character)
Gilles (] )—sometimes Gille—is a stock character of French farce and Commedia dell'Arte.

 He enjoyed his greatest vogue in 18th-century France, in entertainments both at the fairgrounds of the capital and in private and public theaters, though his
origins can be traced back to the 17th century and, possibly, the century previous.

 A "zanni", or comic servant, he is a type of bungling clown, stupid, credulous, and lewd—a character that shares little, problematically, with the sensitive
figure in Watteau's famous portrait that, until the latter half of the 20th century, bore his name alone.

 Gilles fades from view in the 19th century, to persist in the 20th and 21st as the Belgian Gilles of Binche Carnival.
</item 6>

<item 7>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 7>

<item 8>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 8>

<item 9>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 9>

<item 10>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 10>
=== END ITEM id=f3f6b2c71784c57f ===

=== ITEM id=57350376acf1aaf9 ===
Gold sentences and facts:
Sentence 0: The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil War.
  Fact 0.0: The Seven Days Battles were a series of six major battles.
  Fact 0.1: The Seven Days Battles were fought over the seven days from June 25 to July 1, 1862.
  Fact 0.2: The Seven Days Battles were fought near Richmond, Virginia.
  Fact 0.3: The Seven Days Battles took place during the American Civil War.
Sentence 1: The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.
  Fact 1.0: The Battle of Manila took place from February 3, 1945 to March 3, 1945.
  Fact 1.1: The Battle of Manila was a major battle of the Philippine campaign of 1944-45.
  Fact 1.2: The Battle of Manila took place during the Second World War.

Retrieved items:
<item 1>
Seven Days Battles
The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil
War.

 Confederate General Robert E. Lee drove the invading Union Army of the Potomac, commanded by Maj. Gen. George B. McClellan, away from Richmond and into a
retreat down the Virginia Peninsula.

 The series of battles is sometimes known erroneously as the Seven Days Campaign, but it was actually the culmination of the Peninsula Campaign, not a separate
campaign in its own right.
</item 1>

<item 2>
Battle of Manila (1574)
The Battle of Manilla (1574) was a battle in the Manila area mainly in the location of what is now Parañaque between Chinese pirates, led by Limahong and the
Spanish colonial forces and their native allies.

 The battle occurred on November 29, 1574 when Limahong's fleet landed in the town of Parañaque and from there, began to assault the fortifications of
Intramuros.

 Initially, the inhabitants where disorganized and Limahong's forces routed them.

 Furthermore, the Chinese killed the Master-of-Camp of the Spanish, Martin De Goiti.

 This caused them to delay their assault on Manila as Martin de Goiti's house was an obstacle in their march.

 However, upon the arrival of a certain Filipino hero called, Galo, resistance started to organize.

 Under Galo's command, they were able to resist China-born Limahong until Mexico-born Juan de Salcedo arrived from Ilocos with 300 Ilocano Warriors and Limahong
was defeated and eventually forced to retreat.

 Thereafter Limahong abandoned his plans to invade Manila and instead, set up a temporary kingdom in Pangasinan.
</item 2>

<item 3>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 3>

<item 4>
Seven Days Confederate order of battle
The following Confederate States Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization during the battle, the casualty returns and the reports.

 The Union order of battle is listed separately.
</item 4>

<item 5>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 5>

<item 6>
Seven Days Union order of battle
The following Union Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization-return of casualties during the battle and the reports.

 The Confederate order of battle is listed separately.
</item 6>

<item 7>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 7>

<item 8>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 8>

<item 9>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 9>

<item 10>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 10>
=== END ITEM id=57350376acf1aaf9 ===

=== ITEM id=f0014da5e868e7ff ===
Gold sentences and facts:
Sentence 0: Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.
  Fact 0.0: Pietro Alessandro Gaspare Scarlatti was born on 2 May 1660.
  Fact 0.1: Pietro Alessandro Gaspare Scarlatti died on 22 October 1725.
  Fact 0.2: Pietro Alessandro Gaspare Scarlatti was an Italian Baroque composer.
  Fact 0.3: Pietro Alessandro Gaspare Scarlatti was especially famous for his operas.
  Fact 0.4: Pietro Alessandro Gaspare Scarlatti was especially famous for his chamber cantatas.
Sentence 1: Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky, was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.
  Fact 1.0: The Russian name of Pyotr Ilyich Tchaikovsky is Пётр Ильи́ч Чайко́вский.
  Fact 1.1: Pyotr Ilyich Tchaikovsky was born on 25 April/7 May 1840.
  Fact 1.2: Pyotr Ilyich Tchaikovsky died on 25 October/6 November 1893.
  Fact 1.3: Pyotr Ilyich Tchaikovsky is often anglicized as Peter Ilich Tchaikovsky.
  Fact 1.4: Pyotr Ilyich Tchaikovsky was a Russian composer of the romantic period.
  Fact 1.5: Some works by Pyotr Ilyich Tchaikovsky are among the most popular music in the classical repertoire.
Sentence 2:  He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the United States.
  Fact 2.0: Tchaikovsky was the first Russian composer whose music made a lasting impression internationally.
  Fact 2.1: The international impression of Tchaikovsky's music was bolstered by his appearances as a guest conductor.
  Fact 2.2: Tchaikovsky appeared as a guest conductor in Europe.
  Fact 2.3: Tchaikovsky appeared as a guest conductor in the United States.

Retrieved items:
<item 1>
Piano Concerto No. 2 (Tchaikovsky)
Pyotr Ilyich Tchaikovsky's Piano Concerto No. 2 in G major, Op.

 44, was written in 1879–1880.

 It was dedicated to Nikolai Rubinstein, who had insisted he be allowed to perform it at the premiere as a way of making up for his harsh criticism of
Tchaikovsky's First Piano Concerto.

 Rubinstein was destined never to play it, however, as he died in March 1881.

 The premiere performance took place in New York City, on 12 November 1881.

 The soloist was Madeline Schiller, and Theodore Thomas conducted the New York Philharmonic orchestra.

 The first Russian performance was in Moscow in May 1882, conducted by Anton Rubinstein with Tchaikovsky's pupil, Sergei Taneyev, at the piano.
</item 1>

<item 2>
Alessandro Scarlatti
Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.

 He is considered the founder of the Neapolitan school of opera.

 He was the father of two other composers, Domenico Scarlatti and Pietro Filippo Scarlatti.
</item 2>

<item 3>
Kamarinskaya
Kamarinskaya (Russian: камаринская ) is a Russian traditional folk dance, which is mostly known today as the Russian composer Mikhail Glinka's composition of
the same name.

 Glinka's "Kamarinskaya", written in 1848, was the first orchestral work based entirely on Russian folk song and to use the compositional principles of that
genre to dictate the form of the music.

 It became a touchstone for the following generation of Russian composers ranging from the Western-oriented Pyotr Ilyich Tchaikovsky to the group of
nationalists known collectively as The Five and was also lauded abroad, most notably by French composer Hector Berlioz.
</item 3>

<item 4>
Pyotr Ilyich Tchaikovsky
Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky,
was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.

 He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the
United States.

 Tchaikovsky was honored in 1884, by Emperor Alexander III, and awarded a lifetime pension.
</item 4>

<item 5>
Piano Sonata in C-sharp minor (Tchaikovsky)
The Piano Sonata in C-sharp minor, Op.

 posth.

 80, was written by Russian composer Pyotr Ilyich Tchaikovsky in 1865, his last year as a student at the St Petersburg Conservatory.

 The sonata in its original form was not published in Tchaikovsky's lifetime; it was published in 1900 by P. Jurgenson, and given the posthumous opus number 80.
</item 5>

<item 6>
Chaikovskij (crater)
Chaikovskij (sometimes Tchaikovsky) is a crater on Mercury.

 It has a diameter of 165 kilometers.

 Its name was adopted by the International Astronomical Union in 1976.

 Chaikovskij is named for the Russian composer Pyotr Ilyich Tchaikovsky, who lived from 1840 to 1893.
</item 6>

<item 7>
Symphony No. 2 (Rachmaninoff)
Symphony No. 2 in E minor, Op. 27 is a symphony by the Russian composer Sergei Rachmaninoff, written in 1906–07.

 The premiere was conducted by the composer himself in Saint Petersburg on 8 February 1908.

 Its duration is approximately 60 minutes when performed uncut; cut performances can be as short as 35 minutes.

 The score is dedicated to Sergei Taneyev, a Russian composer, teacher, theorist, author, and pupil of Pyotr Ilyich Tchaikovsky.

 Alongside his Piano Concerto No. 2 and Piano Concerto No. 3, this symphony remains one of the composer's best known compositions.
</item 7>

<item 8>
International Tchaikovsky Competition
The International Tchaikovsky Competition is a classical-music competition held every four years in Moscow, Russia, for pianists, violinists, and cellists
between 16 and 32 years of age, and singers between 19 and 32 years of age.

 The competition is named after Russian composer Pyotr Ilyich Tchaikovsky and is an active member of the World Federation of International Music Competitions.
</item 8>

<item 9>
Swan Lake (1895)
The 1917 Petipa/Ivanov/Drigo revival of Swan Lake is a famous version of the ballet "Swan Lake", (ru. "

Лебединое Озеро"), (fr.

 "Le Lac des Cygnes").

 This is a ballet by Pyotr Ilyich Tchaikovsky based on an ancient German legend, presented in either four acts, four scenes (primarily outside Russia and
Eastern Europe), three acts, four scenes (primarily in Russia and Eastern Europe) or, more rarely, in two acts, four scenes.

 Originally choreographed by Julius Reisinger to the music of Pyotr Ilyich Tchaikovsky (opus 20), it was first presented as "The Lake of the Swans" by the
Ballet of the Moscow Imperial Bolshoi Theatre on 20 February/4 March 1877 (Julian/Gregorian calendar dates) in Moscow, Russia.
</item 9>

<item 10>
Tchaikovsky (film)
Tchaikovsky (Russian: Чайковский ) is a 1970 Soviet biopic film directed by Igor Talankin.

 It featured Innokenty Smoktunovsky in the role of the famous Russian composer Pyotr Ilyich Tchaikovsky.

 It was nominated for the 1971 Academy Award for Best Foreign Language Film as well as the Academy Award for Original Song Score and Adaptation.
</item 10>
=== END ITEM id=f0014da5e868e7ff ===

=== ITEM id=0c88598778c00aa8 ===
Gold sentences and facts:
Sentence 0:  It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center near the Miami River.
  Fact 0.0: It flows in a west to east direction.
  Fact 0.1: It flows from the western part of the state in the Everglades.
  Fact 0.2: It flows past the Miami International Airport.
  Fact 0.3: It flows to a salinity control center near the Miami River.
Sentence 1: The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
  Fact 1.0: The Genesee River is a tributary of Lake Ontario.
  Fact 1.1: The Genesee River flows northward.
  Fact 1.2: The Genesee River flows through the Twin Tiers of Pennsylvania.
  Fact 1.3: The Genesee River flows through the Twin Tiers of New York in the United States.

Retrieved items:
<item 1>
Tamiami Canal
The Tamiami Canal or C-4 Canal, is a canal located in southern Florida in the United States.

 It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center
near the Miami River.

 It averages 8 ft in depth and is over 100 ft wide in some areas.
</item 1>

<item 2>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 2>

<item 3>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 3>

<item 4>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 4>

<item 5>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 5>

<item 6>
Oatka Creek
Oatka Creek ( ) is the third longest tributary of the Genesee River, located entirely in the Western New York region of the U.S. state of New York.

 From southern Wyoming County, it flows 58 mi to the Genesee near Scottsville, draining an area of 215 sqmi that includes all or part of 23 towns and villages
in Wyoming, Genesee, Livingston and Monroe counties as well.

 Its name means "leaving the highlands" or "approaching an opening" in Seneca.
</item 6>

<item 7>
Genesee River
The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
</item 7>

<item 8>
Genesee Valley Park
Genesee Valley Park is located in the south side of Rochester, New York along the shores of the Genesee River.

 The New York State Barge Canal (the currently in-use portion of the Erie Canal) crosses the Genesee River within the park.

 The University of Rochester is located near the park's north entrance.
</item 8>

<item 9>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 9>

<item 10>
Pennsylvania Canal (Delaware Division)
The Delaware Division of the Pennsylvania Canal, more commonly called the Delaware Canal, runs for 60 mi parallel to the right bank of the Delaware River from
the entry locks near the mouth of the Lehigh River and terminal end of the Lehigh Canal at Easton south to Bristol.

 At Easton, which today is the home of The National Canal Museum, the Delaware Canal also connected with the Morris Canal built to carry anthracite coal to
energy starved New Jersey industries.

 Later, with a crossing-lock constructed at New Hope, the New Hope 'outlet lock' (1847) connected by Cable Ferry to a feeder navigation/canal at Bull's Island
along the opposite shore from Lumberville, which ran over 22 mi south along the left bank through Trenton to Bordentown, the west end of the Delaware and
Raritan Canal (1834) to New York City via New Brunswick.

 as part of the solution to the United States' first energy crisis.
</item 10>
=== END ITEM id=0c88598778c00aa8 ===

=== ITEM id=b0bfca49a3c85b53 ===
Gold sentences and facts:
Sentence 0:  Related fields of study include Islamic philosophy and Sufism.
  Fact 0.0: Related fields of study include Islamic philosophy.
  Fact 0.1: Related fields of study include Sufism.
Sentence 1: Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular] values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most important and central crystallization of" mystical practice in Islam.
  Fact 1.0: Sufism is also called Taṣawwuf.
  Fact 1.1: The Arabic name for Sufism is الْتَّصَوُّف [sect].
  Fact 1.2: The personal noun for a Sufi is صُوفِيّ, transliterated as 'ṣūfiyy' or 'ṣūfī'.
  Fact 1.3: The personal noun mutaṣawwūf is مُتَصَوّف.
  Fact 1.4: Sufism is often defined as 'Islamic mysticism'.
  Fact 1.5: Sufism is often defined as 'the inward dimension of Islam'.
  Fact 1.6: Sufism is often defined as 'the phenomenon of mysticism within Islam'.
  Fact 1.7: Sufism is a mystical trend in Islam.
  Fact 1.8: Sufism is characterized by particular values, ritual practices, doctrines and institutions.
  Fact 1.9: Sufism began very early on in Islamic history.
  Fact 1.10: Sufism represents the main manifestation of mystical practice in Islam.
  Fact 1.11: Sufism represents the most important and central crystallization of mystical practice in Islam.

Retrieved items:
<item 1>
Reynold A. Nicholson
Reynold Alleyne Nicholson, or R. A. Nicholson (18 August 1868 – 27 August 1945), was an eminent English orientalist, scholar of both Islamic literature and
Islamic mysticism and widely regarded as one of the greatest Rumi (Mevlana or Mawlana) scholars and translators in the English language.
</item 1>

<item 2>
Dawud Tai
Abu Solaiman Dawud ibn Nosair al-Ta’i, usually referred to as Dawud Tai, (died between 777 and 782) was a Islamic scholar and Sufi mystic.

 His disciples included many influential personalities of Islamic mysticism, e.g., Maruf Karkhi and Ahmad al-Buni.
</item 2>

<item 3>
Ovamir Anjum
Ovamir Anjum is the Imam Khattab Chair of Islamic Studies at the Department of Philosophy, University of Toledo.

 He studies the connections between theology, ethics, politics, and law in classical and medieval Islam, with a subfocus on its comparisons with western
thought.

 Related fields of study include Islamic philosophy and Sufism.
</item 3>

<item 4>
Animality studies
Animality studies is an emerging interdisciplinary academic field focused on the cultural study of animals and animality.

 It can be distinguished from animal studies and critical animal studies by its resistance to animal rights or animal welfare as an explicit justification for
work in this field.

 According to Michael Lundblad, "If animal studies can be seen as work that explores representations of animality and related discourses with an emphasis on
advocacy for nonhuman animals, animality studies becomes work that emphasizes the history of animality in relation to human cultural studies, without an
explicit call for nonhuman advocacy."
</item 4>

<item 5>
Omid Safi
Omid Safi is an American Professor of Asian and Middle Eastern Studies at Duke University, where he is the Director of Duke Islamic Studies Center, and a
columnist for On Being.

 Dr. Safi specializes in Islamic mysticism (Sufism), contemporary Islamic thought and medieval Islamic history.

 He has served on the board of the Pluralism project at Harvard University and is the co-chair of the steering committee for the Study of Islam and the Islamic
Mysticism Group at the American Academy of Religion.

 Before joining Duke University, Dr. Safi was a professor at the University of North Carolina Chapel Hill.
</item 5>

<item 6>
Irfan
In Islam, ‘Irfaan (Arabic/Persian/Urdu: عرفان ; Turkish: "İrfan" ), also spelt "Irfaan" and "Erfan", literally ‘knowledge, awareness, wisdom’, is gnosis.

 Islamic mysticism can be considered as a vast range that engulfs theoretical and practical and conventional mysticism and has been intertwined with sufism and
in some cases they are assumed identical.

 however Islamic mysticism is assumed as one of the Islamic sciences alongside theology and philosophy.

 Islamic’s mysticism is cognition and knowledge that love has been intertwined through it with structure of revelation in Islam.
</item 6>

<item 7>
Madonna Studies
Madonna Studies is a term which has been used to refer to a development of a field in several studios since late 1980s.

 One writer described the "institutionalization of a major subdivision of American media studies into Madonna studies".

 The field related to the study of and interpretation of the work of American pop musician Madonna using an interdisciplinary approach incorporating cultural
studies and media studies.

 A notable compendium of essays titled "The Madonna Connection" was published in 1992.

 Controversy over this field of study stemmed from discussions over the intellectual worth of pursuing academic inquiry into a pop musician, with some arguing
the field was nothing more than pop cultural commentary.
</item 7>

<item 8>
A Hidden Treasure
Hidden Treasure (Arabic: کنزاً مخفیاً‎ ‎ ) is a Hadith Qudsi that has a very prominent role in Islamic mysticism and Islamic philosophy.
</item 8>

<item 9>
Sufism
Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic
mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular]
values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most
important and central crystallization of" mystical practice in Islam.

 Although the overwhelming majority of Sufis, both pre-modern and modern, have been adherents of Sunni Islam, there nevertheless also developed certain strands
of Sufi practice within the ambit of Shia Islam during the late medieval period.
</item 9>

<item 10>
Moinuddin Chishti
1325) and Amir Khusrow (d. 1325).

 As such, Muʿīn al-Dīn Chishtī's legacy rests primarily on his having been "one of the most outstanding figures in the annals of Islamic mysticism."

 Additionally, Muʿīn al-Dīn Chishtī is also notable for having been one of the first major Islamic mystics to formally allow his followers to incorporate the
"use of music" in their devotions, liturgies, and hymns to God, which he did so in order to make the foreign Arab faith more relatable to the indigenous peoples
who had recently entered the religion or whom he sought to convert.
</item 10>
=== END ITEM id=b0bfca49a3c85b53 ===

=== ITEM id=b9a41b045dedad22 ===
Gold sentences and facts:
Sentence 0: John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).
  Fact 0.0: John H. Garabedian is an American radio personality.
  Fact 0.1: John H. Garabedian is an American disc jockey.
  Fact 0.2: John H. Garabedian was born on December 20, 1941.
  Fact 0.3: John H. Garabedian is age 75.
Sentence 1: Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
  Fact 1.0: Charles A. Agemian was born July 22, 1909.
  Fact 1.1: Charles A. Agemian's life ended March 30, 1996.
  Fact 1.2: Charles A. Agemian was an American banker.
  Fact 1.3: Charles A. Agemian took early retirement from Chase Manhattan Bank.
  Fact 1.4: Charles A. Agemian was executive vice president of operations at Chase Manhattan Bank.
  Fact 1.5: Charles A. Agemian became chairman of the Hackensack Trust Co.
  Fact 1.6: Charles A. Agemian became chief executive officer of the Hackensack Trust Co.
  Fact 1.7: The Hackensack Trust Co. was later renamed Garden State National Bank.

Retrieved items:
<item 1>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 1>

<item 2>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 2>

<item 3>
Radio personality
A radio personality (American English) or radio presenter (British English), commonly referred to as a "disc jockey" or "DJ" for short, is a person who has an
on-air position in radio broadcasting.

 A radio personality that hosts a radio show is also known as a radio host, and in India and Pakistan as a radio jockey.

 Radio personalities who introduce and play individual selections of recorded music are known as disc jockeys.

 The term has evolved to also describe a person who mixes a continuous flow of recorded music in real time.

 Broadcast radio personalities may include talk radio hosts, AM/FM radio show hosts, and satellite radio program hosts.

 Notable radio personalities include pop music radio hosts Martin Block, Alan Freed, Dick Clark, Delilah Luke, Ameen Sayani, Wolfman Jack, and Casey Kasem,
shock jocks such as Don Imus and Howard Stern, as well as sports talk hosts such as Mike Francesa and political talk hosts such as Rush Limbaugh.
</item 3>

<item 4>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 4>

<item 5>
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 5>

<item 6>
Don Sherwood (DJ)
Don Sherwood (September 7, 1925 – November 6, 1983) was an American radio personality.

 He was a San Francisco, California, disc jockey during the 1950s and 1960s.

 Billed as "The World's Greatest Disc Jockey," Sherwood spent most of his career hosting a 6-9 a.m. weekday program on KSFO in San Francisco (560 kHz, 5000
watts), which was then owned by the singing cowboy actor Gene Autry.
</item 6>

<item 7>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 7>

<item 8>
M. G. Kelly
Gary Sinclair (born 1952 in Ada, Oklahoma), known professionally as Michael Gary “M.G.” Kelly and Machine Gun Kelly, is an American actor, disc jockey and radio
personality.

 In addition to hosting several radio programs over the years, Kelly has held several acting roles as a disc jockey; also, he has served as an offstage
announcer on two game shows.
</item 8>

<item 9>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
</item 9>

<item 10>
Big Wilson
Big Wilson (born Malcolm John Wilson, Jr.; October 3, 1924 – October 5, 1989) was an American radio personality.

 He worked as a disc jockey at WNBC AM in New York City from the early 1960s until 1974 and moved to Miami in 1975 where he worked for WIOD and WCIX-TV.

 He was one of the last two hosts of the NBC network program Monitor.

 Prior to moving to NYC, Wilson and his wife Jody lived in a riverfront home in Rocky River, Ohio.

 He commuted to Cleveland where he was the leading radio personality for some years.

 Tim Conway and other well-known persons visited their home.
</item 10>
=== END ITEM id=b9a41b045dedad22 ===
