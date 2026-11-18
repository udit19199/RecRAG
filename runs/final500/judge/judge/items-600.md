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

=== ITEM id=1825dfae2f98b13e ===
Gold sentences and facts:
Sentence 0: MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen neighborhood of Manhattan, New York City.
  Fact 0.0: MiMA stands for 'Middle of Manhattan'.
  Fact 0.1: MiMA is a mixed-use building.
  Fact 0.2: MiMA is located at 450 West 42nd Street.
  Fact 0.3: MiMA is located between Dyer and 10th Avenues.
  Fact 0.4: MiMA is located in the Hell's Kitchen neighborhood of Manhattan, New York City.
Sentence 1:  Ground was broken in 2007 and topping out occurred in early August 2010.
  Fact 1.0: Ground was broken on MiMA in 2007.
  Fact 1.1: Topping out of MiMA occurred in early August 2010.
Sentence 2:  It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on floors 51 to 63, and a Yotel hotel on the lower levels.
  Fact 2.0: MiMA was designed by the Miami-based architecture firm of Arquitectonica.
  Fact 2.1: MiMA has 43 floors of luxury rentals on floors 7 to 50.
  Fact 2.2: MiMA has twelve floors of condominiums on floors 51 to 63.
  Fact 2.3: MiMA has a Yotel hotel on the lower levels.
Sentence 3:  The building's height is 204 meters (670 feet).
  Fact 3.0: The height of MiMA is 204 meters (670 feet).
Sentence 4: 270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan, New York City.
  Fact 4.0: 270 Park Avenue is also known as the JPMorgan Chase Tower.
  Fact 4.1: 270 Park Avenue was formerly known as the Union Carbide Building.
  Fact 4.2: 270 Park Avenue is a high-rise office building.
  Fact 4.3: 270 Park Avenue is located in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 1>

<item 2>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 2>

<item 3>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 3>

<item 4>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 4>

<item 5>
Park Avenue House
The Park Avenue House is a high rise residential building located at 2305 Park Avenue in the Park Avenue Historic District in Downtown Detroit, Michigan.

 It was listed on the National Register of Historic Places in 1996.

 It should not be confused with the nearby Park Avenue Hotel, which was demolished in 2015.
</item 5>

<item 6>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 6>

<item 7>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 7>

<item 8>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 8>

<item 9>
MiMA (building)
MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen
neighborhood of Manhattan, New York City.

 Ground was broken in 2007 and topping out occurred in early August 2010.

 It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on
floors 51 to 63, and a Yotel hotel on the lower levels.

 The building's height is 204 meters (670 feet).
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
=== END ITEM id=1825dfae2f98b13e ===

=== ITEM id=2664a792b46485d0 ===
Gold sentences and facts:
Sentence 0: William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story writer.
  Fact 0.0: William Somerset Maugham was born 25 January 1874.
  Fact 0.1: William Somerset Maugham died 16 December 1965.
  Fact 0.2: William Somerset Maugham is better known as W. Somerset Maugham.
  Fact 0.3: William Somerset Maugham was British.
  Fact 0.4: William Somerset Maugham was a playwright.
  Fact 0.5: William Somerset Maugham was a novelist.
  Fact 0.6: William Somerset Maugham was a short story writer.
  Fact 0.7: William Somerset Maugham's name includes the post-nominal CH.
Sentence 1:  He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
  Fact 1.0: W. Somerset Maugham was among the most popular writers of his era.
  Fact 1.1: W. Somerset Maugham was reputedly the highest-paid author during the 1930s.
Sentence 2: Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.
  Fact 2.0: Thomas Merton, O.C.S.O., was born January 31, 1915.
  Fact 2.1: Thomas Merton died December 10, 1968.
  Fact 2.2: Thomas Merton was an American Catholic writer.
  Fact 2.3: Thomas Merton was a theologian.
  Fact 2.4: Thomas Merton was a mystic.
Sentence 3:  In 1949, he was ordained to the priesthood and given the name Father Louis.
  Fact 3.0: In 1949, Thomas Merton was ordained to the priesthood.
  Fact 3.1: Thomas Merton was given the name Father Louis.

Retrieved items:
<item 1>
Trio (film)
Trio (also known as W. Somerset Maugham's Trio) is a 1950 British anthology film based on three short stories by W. Somerset Maugham: "The Verger", "Mr.
Know-All" and "Sanatorium".

 Ken Annakin directed "The Verger" and "Mr. Know-All", while Harold French was responsible for "Sanatorium".
</item 1>

<item 2>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 2>

<item 3>
W. Somerset Maugham
William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story
writer.

 He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
</item 3>

<item 4>
Somerset Maugham Award
The Somerset Maugham Award is a British literary prize given each May by the Society of Authors.

 It is awarded to the best writer or writers of a book published in the past year who is under the age of 35.

 The prize was instituted in 1947 by William Somerset Maugham and thus bears his name.

 The award is to be spent on foreign travel.

 The total fund for each year is £12,000.
</item 4>

<item 5>
Thomas Merton
Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.

 A Trappist monk of the Abbey of Gethsemani, Kentucky, he was a poet, social activist, and student of comparative religion.

 In 1949, he was ordained to the priesthood and given the name Father Louis.
</item 5>

<item 6>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 6>

<item 7>
Charles Leslie Wrenn
Charles Leslie Wrenn (1895–1969) was a British scholar.

 He became Rawlinson and Bosworth Professor of Anglo-Saxon at the University of Oxford in 1945, the successor in the chair of J.R.R. Tolkien, and held the
position until 1963.

 Wrenn was a Fellow of Pembroke College, Oxford.

 He was also a member of the Oxford literary discussion group known as the "Inklings", which included C. S. Lewis and Tolkien.

 Some of the work published by Wrenn includes "The English Language" (1949), "A Study of Old English Literature" (1967)"," and "An Old English Grammar," written
with Randolph Quirk (1955, rev. 1957).

 His literary interests were primarily comparative literature and later poets including T. S. Elliot.
</item 7>

<item 8>
List of works about Thomas Merton
The following is a list of works about Thomas Merton, publications about Thomas Merton, the Trappist monk from Abbey of Gethsemani.

 The bibliography is organized into categories.

 A separate list of works by Thomas Merton is also available.
</item 8>

<item 9>
St. Joseph's Intermediate and Commercial School
St. Joseph's Intermediate and Commercial School (St. Joseph's Commercial School, SJICS, or St. Joe's Sherbourne) is a former Catholic High school located in the
Cabbagetown neighbourhood of Toronto, Ontario, Canada part of the Metropolitan Separate School Board (now the Toronto Catholic District School Board).

 The school was later renamed to Thomas Merton Academy (Thomas Merton, TMA, or Merton) in 1985 after the Anglo-American monk Thomas Merton.
</item 9>

<item 10>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 10>
=== END ITEM id=2664a792b46485d0 ===

=== ITEM id=a88694b30d63558d ===
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
Graph query row:
person: Alessandro Scarlatti
nationality: Italian
occupation: composer
birth_date: 2 May 1660
death_date: 22 October 1725
relationship: CHILD_OF
connected_entity: Domenico Scarlatti
source_title: Alessandro Scarlatti
</item 1>

<item 2>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 2>

<item 3>
Graph query row:
person: Pyotr Ilyich Tchaikovsky
nationality: Russian
occupation: composer
birth_date: None
death_date: None
relationship: CREATED_BY
connected_entity: Piano Concerto No. 2
source_title: Chaikovskij (crater)
</item 3>

<item 4>
Piano Sonata in C-sharp minor (Tchaikovsky)
The Piano Sonata in C-sharp minor, Op.

 posth.

 80, was written by Russian composer Pyotr Ilyich Tchaikovsky in 1865, his last year as a student at the St Petersburg Conservatory.

 The sonata in its original form was not published in Tchaikovsky's lifetime; it was published in 1900 by P. Jurgenson, and given the posthumous opus number 80.
</item 4>

<item 5>
Pyotr Ilyich Tchaikovsky
Pyotr Ilyich Tchaikovsky ( ; Russian: Пётр Ильи́ч Чайко́вский ; 25 April/7 May 1840 – 25 October/6 November 1893), often anglicized as Peter Ilich Tchaikovsky,
was a Russian composer of the romantic period, some of whose works are among the most popular music in the classical repertoire.

 He was the first Russian composer whose music made a lasting impression internationally, bolstered by his appearances as a guest conductor in Europe and the
United States.

 Tchaikovsky was honored in 1884, by Emperor Alexander III, and awarded a lifetime pension.
</item 5>

<item 6>
Tchaikovsky (film)
Tchaikovsky (Russian: Чайковский ) is a 1970 Soviet biopic film directed by Igor Talankin.

 It featured Innokenty Smoktunovsky in the role of the famous Russian composer Pyotr Ilyich Tchaikovsky.

 It was nominated for the 1971 Academy Award for Best Foreign Language Film as well as the Academy Award for Original Song Score and Adaptation.
</item 6>

<item 7>
International Tchaikovsky Competition
The International Tchaikovsky Competition is a classical-music competition held every four years in Moscow, Russia, for pianists, violinists, and cellists
between 16 and 32 years of age, and singers between 19 and 32 years of age.

 The competition is named after Russian composer Pyotr Ilyich Tchaikovsky and is an active member of the World Federation of International Music Competitions.
</item 7>

<item 8>
Alessandro Scarlatti
Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.

 He is considered the founder of the Neapolitan school of opera.

 He was the father of two other composers, Domenico Scarlatti and Pietro Filippo Scarlatti.
</item 8>

<item 9>
Kamarinskaya
Kamarinskaya (Russian: камаринская ) is a Russian traditional folk dance, which is mostly known today as the Russian composer Mikhail Glinka's composition of
the same name.

 Glinka's "Kamarinskaya", written in 1848, was the first orchestral work based entirely on Russian folk song and to use the compositional principles of that
genre to dictate the form of the music.

 It became a touchstone for the following generation of Russian composers ranging from the Western-oriented Pyotr Ilyich Tchaikovsky to the group of
nationalists known collectively as The Five and was also lauded abroad, most notably by French composer Hector Berlioz.
</item 9>

<item 10>
Piano Concerto No. 2 (Tchaikovsky)
Pyotr Ilyich Tchaikovsky's Piano Concerto No. 2 in G major, Op.

 44, was written in 1879–1880.

 It was dedicated to Nikolai Rubinstein, who had insisted he be allowed to perform it at the premiere as a way of making up for his harsh criticism of
Tchaikovsky's First Piano Concerto.

 Rubinstein was destined never to play it, however, as he died in March 1881.

 The premiere performance took place in New York City, on 12 November 1881.

 The soloist was Madeline Schiller, and Theodore Thomas conducted the New York Philharmonic orchestra.

 The first Russian performance was in Moscow in May 1882, conducted by Anton Rubinstein with Tchaikovsky's pupil, Sergei Taneyev, at the piano.
</item 10>
=== END ITEM id=a88694b30d63558d ===

=== ITEM id=d52d9d21ea4eb8d1 ===
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
Love Incredible
"Love Incredible" is a song by Norwegian DJ and record producer Cashmere Cat, featuring vocals by Cuban-American singer Camila Cabello.

 It was released on 17 February 2017 through Interscope and Mad Love as the third single from his debut album, "9".
</item 1>

<item 2>
Gallo Record Company
Gallo Record Company is the largest (and oldest independent) record label in South Africa.

 It is based in Johannesburg, South Africa, and is owned by Times Media Group (formerly Johnnic Communications and Avusa).

 The current Gallo Record Company is a hybrid of two rival South African record labels between the 1940s and 1980s: the original Gallo Africa (1926–85) and
G.R.C. (Gramophone Record Company, 1939–85).

 In 1985 Gallo Africa acquired G.R.C.; as a result, Gallo Africa became known as "Gallo-GRC".

 Five years after the acquisition, the company was renamed "Gallo Record Company".
</item 2>

<item 3>
Recess Records
Recess Records is an independent record label founded in 1989 with the release of F.Y.P's "Extra Credit", an 11 song 7-inch record.

 Label founder Todd Congelliere (frontman for F.Y.P and Toys That Kill) originally created the label as a vehicle to release F.Y.P's material.

 Congelliere promoted "Extra Credit" by running classified ads in Berkeley based Maximum Rocknroll fanzine and going to skateboard contests.

 He quickly sold his initial pressing of 500 copies.

 In between touring the U.S., Europe, and Japan, Congelliere began releasing records for fellow punk rock and DIY bands, eventually having a roster of close to
20 artists on his label.

 Pinhead Gunpowder, fronted by Green Day frontman Billie Joe Armstrong, announced in April 2008 that a new 7" single from the band would be released on Recess
Records in late May 2008.
</item 3>

<item 4>
Interscope Records
Interscope Records is an American major record label.

 An imprint of Interscope Geffen A&M Records, its parent company is Universal Music Group, a subsidiary of Vivendi S.A.
</item 4>

<item 5>
Incredible Connection
Incredible Connection (Pty) Ltd is the largest consumer electronics and IT retailer in South Africa.

 On 15 December 1998, Incredible Connection was acquired by Connection Group Holdings Limited and now operates as subsidiary of that company, itself a
subsidiary of JD Group Limited.

 As of 2005, Incredible Connection has 34 retail outlets in South Africa, and in 2002 it was reported to hold 45% of the IT retail market in the country.
</item 5>

<item 6>
Kinetic Records
The label was distributed by Reprise/Warner Bros.

 Records from its inception until 2001, when Kinetic switched distribution to BMG.

 The latter eventually absorbed the label during the merger with Sony Music.
</item 6>

<item 7>
American Record Corporation
American Record Corporation (ARC), also referred to as American Record Company, American Recording Corporation, or (erroneously) as ARC Records, was an American
record company.

 It resulted from the merger in 1929 of three companies: the Cameo Record Corporation (which owned Cameo, Lincoln and Romeo Records), the Pathé Phonograph and
Radio Corporation (which owned Actuelle, Pathé, and Perfect), and the Plaza Music Company (which owned Banner, Domino, Jewel, Oriole, and Regal).
</item 7>

<item 8>
Manimal Vinyl
Manimal Vinyl is a Calabasas based record label founded in 2006 by experimental musician and fashion stylist, Paul Beahan.

 They were originally known as a psychedelic folk label that makes semi-annual tribute records for worldwide charities, having eventually switched to dance
music and progressive underground.

 They are the label who has release debut records from act ranging from Warpaint to Bat for Lashes and exclusive singles from Moby, Duran Duran, Carla Bruni,
Edward Sharpe and the Magnetic Zeros.

 In 2008, they founded the annual Manimal Festival in Pioneertown, California which ran from 2008-2010.

 The festival has featured line-ups with Edward Sharpe and the Magnetic Zeros, Warpaint, Ariel Pink and papercranes.

 Manimal has offices in Los Angeles and New York City.
</item 8>

<item 9>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=d52d9d21ea4eb8d1 ===

=== ITEM id=510e0bc807c37b18 ===
Gold sentences and facts:
Sentence 0:  He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.
  Fact 0.0: He won three singles titles.
  Fact 0.1: He won 26 doubles titles.
  Fact 0.2: He won the Australian Open 1978 doubles title with Wojtek Fibak.
  Fact 0.3: He won the Australian Open 1980 title with fellow countryman Mark Edmondson.
  Fact 0.4: He won the Australian Open 1981 title with fellow countryman Mark Edmondson.
  Fact 0.5: He won the Roland Garros 1986 title with fellow countryman Mark Edmondson.
  Fact 0.6: He was a runner-up at the Australian Open 1985 with fellow countryman Mark Edmondson.
  Fact 0.7: Mark Edmondson is a fellow countryman of his.
Sentence 1:  In his career, he won nine doubles titles, but no singles titles.
  Fact 1.0: In his career, he won nine doubles titles.
  Fact 1.1: In his career, he won no singles titles.

Retrieved items:
<item 1>
Mexican runner Nora Rocha won the 10,000m gold at the 1999 Pan American Games. Paraguayan tennis player Rossana de los Ríos reached No. 51 in singles, won the
1992 French Open junior title, and represented Paraguay at two Olympics. Argentine Paola Suárez won eight Grand Slam doubles titles with Virginia Ruano Pascual
and reached the 2004 French Open singles semifinals; in 1999 she won the Brasil
</item 1>

<item 2>
Jorge Lozano
In Davis Cup play, represented Mexico for 15 years, 1981–1995, won 12 doubles matches, and lost 12 as well.

 He lost 11 singles matches, and won 8, making his overall win / loss record at the Davis Cup 20–23.

 Lozano won his first title at Forest Hills in 1988, with his partner Todd Witsken, and won his last at Athens in 1993.

 He resides in Guadalajara, Mexico where he directs a tennis academy for kids and also coaches the men and women tennis team at the University Tec de Monterrey.

 He is currently the captain of the Mexican Davis Cup Team.
</item 2>

<item 3>
1988 Volvo International – Doubles
Jorge Lozano and Todd Witsken won in the final 6–3, 7–6 against Pieter Aldrich and Danie Visser.
</item 3>

<item 4>
The material compiles tennis results and player biographies. Sherwood Stewart and Ferdi Taygan won the 1982 French Open doubles and finished runner-up at the
Volvo Masters and U.S. Pro Indoor; Stewart later partnered others. Jorge Lozano, a Mexican doubles No. 4, won nine doubles titles, including two French Open
mixed doubles crowns, and captained Mexico’s Davis Cup team. Brazilian Carlos Fernandes reached the 1961 French Open quarterfinals. Argentine Javier Fr
</item 4>

<item 5>
1994 Athens International – Doubles
Horacio de la Peña and Jorge Lozano were the defending champions, but did not participate this year.
</item 5>

<item 6>
Kavita Tungar holds India’s 10-km and half-marathon records and won 2010 Commonwealth bronze and Asian Games silver in the 10,000m. Tennis profiles cover Kim
Warwick’s Australian Open final and doubles titles; Evonne Goolagong’s seven Grand Slam singles titles and former No. 1 ranking; and achievements by Ivanovic,
Janković, Stosur, Safina, Azarenka, Watson and Hlaváčková
</item 6>

<item 7>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 7>

<item 8>
1989 Volvo International – Doubles
Jorge Lozano and Todd Witsken were the defending champions but did not compete that year.
</item 8>

<item 9>
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
</item 9>

<item 10>
1990 Stockholm Open – Doubles
Jorge Lozano and Todd Witsken were the defending champions, but lost in the quarterfinals this year.
</item 10>
=== END ITEM id=510e0bc807c37b18 ===

=== ITEM id=02909b844ebcd34d ===
Gold sentences and facts:
Sentence 0: Brunswick Golden Isles Airport (IATA: BQK, ICAO: KBQK, FAA LID: BQK) , previously known as Glynco Jetport, is a county-owned public-use airport located five nautical miles (9 km) north of the central business district of Brunswick, a city in Glynn County, Georgia, United States.
  Fact 0.0: Brunswick Golden Isles Airport has IATA code BQK.
  Fact 0.1: Brunswick Golden Isles Airport has ICAO code KBQK.
  Fact 0.2: Brunswick Golden Isles Airport has FAA LID BQK.
  Fact 0.3: Brunswick Golden Isles Airport was previously known as Glynco Jetport.
  Fact 0.4: Brunswick Golden Isles Airport is a county-owned public-use airport.
  Fact 0.5: Brunswick Golden Isles Airport is located five nautical miles (9 km) north of the central business district of Brunswick.
  Fact 0.6: Brunswick is a city in Glynn County, Georgia, United States.
Sentence 1: Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana, United States.
  Fact 1.0: Lafayette Regional Airport has IATA code LFT.
  Fact 1.1: Lafayette Regional Airport has ICAO code KLFT.
  Fact 1.2: Lafayette Regional Airport has FAA LID LFT.
  Fact 1.3: Lafayette Regional Airport is a public airport.
  Fact 1.4: Lafayette Regional Airport is two miles (4 km) southeast of Lafayette.
  Fact 1.5: Lafayette Regional Airport is in Lafayette Parish, Louisiana, United States.

Retrieved items:
<item 1>
College of Coastal Georgia
The College of Coastal Georgia (also known as CCGA) is a public college located at historical Brunswick, Georgia, United States and the surrounding historical
Golden Isles.

 The College was established in 1961 and opened in 1964, making it one of Georgia's newest state colleges.

 The college transitioned from a two-year community college into a four-year comprehensive institution and conferred its first baccalaureate degrees on May 7,
2011.
</item 1>

<item 2>
Brunswick Golden Isles Airport
Brunswick Golden Isles Airport (IATA: BQK, ICAO: KBQK, FAA LID: BQK) , previously known as Glynco Jetport, is a county-owned public-use airport located five
nautical miles (9 km) north of the central business district of Brunswick, a city in Glynn County, Georgia, United States.

 It is mostly used for general aviation, but is also served by one commercial airline.
</item 2>

<item 3>
Golden Isles of Georgia
The Golden Isles of Georgia are a group of four barrier islands and the mainland port city of Brunswick on the 100-mile-long coast of the U.S. state of Georgia
on the Atlantic Ocean.

 They include St. Simons Island, Sea Island, Jekyll Island, Little St. Simons Island, and Historic Brunswick.
</item 3>

<item 4>
Jekyll Island Airport
Jekyll Island Airport is a small general aviation airport located in Jekyll Island, GA.

 It is around 2 miles away from central Jekyll Island, GA.

 Due to its small runway size and the nearby Brunswick Golden Isles Airport, no commercial airlines fly there, but scenic tours of the island fly from there.

 The airport has 3 based aircraft, and around 74 aircraft operations a day.
</item 4>

<item 5>
Colonel's Island Railroad
In the late 1960s, the State of Georgia started an improvement project for the port at Brunswick, Georgia.

 In this project, the Colonel's Island Railroad (reporting mark CISD) was created as a part of the Georgia Ports Authority.

 The CISD operated 33 miles of track in and around the Brunswick area.

 In 1998, the railroad sold most of its assets and the Golden Isles Terminal Railroad began operating on the line.

 The Ports Authority retained ownership of the track, but the CISD was disbanded.
</item 5>

<item 6>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 6>

<item 7>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 7>

<item 8>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 8>

<item 9>
Atlanta South Regional Airport
Henry County Airport (FAA LID: KHMP) , is a public-use county airport located three nautical miles (6 km) west of the central business district of Hampton, a
city in Henry County, Georgia, United States.

 It was known as Clayton County Airport – Tara Field, which was the name still used by the Federal Aviation Administration (FAA) and the Georgia Department of
Transportation (GDOT) until approval of the Atlanta South Regional Airport name.

 The airport was renamed Atlanta South Regional Airport, which was approved by the airport board of commissioners in December 2011, and was approved by the GDOT
and the FAA before it went into effect.
</item 9>

<item 10>
Brunswick Old Town Historic District
Brunswick Old Town Historic District is a historic district in Brunswick, Georgia.

 It was added to the National Register of Historic Places on April 26, 1979 and includes an area bounded by 1st Street, Bay Street, New Bay Street, H Street,
and Cochran Street (4th Ave. and G St., according to one source).

 Brunswick is one of Georgia's two deep-water ports and is the mainland city associated with the Golden Isles of Georgia, at the junction of I-95 and US 82.
</item 10>
=== END ITEM id=02909b844ebcd34d ===

=== ITEM id=73723b6b07f44b45 ===
Gold sentences and facts:
Sentence 0:  She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.
  Fact 0.0: The woman described is married to Scott Stapp.
  Fact 0.1: Scott Stapp is a former frontman of the band Creed.
  Fact 0.2: Scott Stapp is the current frontman for the band Art of Anarchy.
Sentence 1:  The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between members.
  Fact 1.0: The band's third album, “Weathered”, was released in 2001.
  Fact 1.1: Tremonti handled bass when “Weathered” was released.
  Fact 1.2: The band disbanded in 2004.
  Fact 1.3: The band disbanded in 2004 due to increasing tension between members.

Retrieved items:
<item 1>
Jaclyn Stapp
Jaclyn Nesheiwat Stapp (born July 29, 1980) is a beauty queen, philanthropist and fashion model with pageant roots in Florida and New York.

 She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.

 Her most notable titles include Mrs. Florida America 2008 and Miss New York USA 2004.

 She is executive director of The Scott Stapp With Arms Wide Open Foundation.
</item 1>

<item 2>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 2>

<item 3>
The Preytells
The Preytells were an Australian indie rock band from Perth, formed in 2004 by Jessica Bennett (a.k.a. Audrey Tell) on lead guitar and backing vocals, Simon
Okely (a.k.a. Will Tell) on guitar and lead vocals, Jaclyn Pearson on drums and percussion and Cameron Stewart on bass guitar and backing vocals.

 Their sole album, "Flood Songs/June Songs", was issued in September 2009; before they disbanded later that year.

 Their single, "Shout!"

 ( May 2008), was nominated for WAM Song of the Year in the Pop category in 2008; while, "Lord Hold My Hand", was nominated for the same category in 2009.
</item 3>

<item 4>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 4>

<item 5>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 5>

<item 6>
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 6>

<item 7>
The Replacements (band)
The Replacements were an American rock band formed in Minneapolis, Minnesota, in 1979.

 Initially a punk rock band, they are considered pioneers of alternative rock.

 The band was composed of the guitarist and vocalist Paul Westerberg, guitarist Bob Stinson, bass guitarist Tommy Stinson and drummer Chris Mars for most of its
career.

 Following several acclaimed albums, including "Let It Be" and "Tim", Bob Stinson left the band in 1986, and Slim Dunlap joined as lead guitarist.

 Steve Foley replaced Mars in 1990.

 Towards the end of the band's career, Westerberg exerted more control over the creative output.

 The group disbanded in 1991, with the members eventually pursuing various projects.

 A reunion was announced on October 3, 2012.

 The Replacements never had significant commercial success, except for "I'll Be You", which hit number 1 on the "Billboard" Alt Rock chart, but they influenced
numerous alternative rock acts.
</item 7>

<item 8>
Creed (band)
Creed is an American rock band formed in 1993 in Tallahassee, Florida.

 The band's best-known line-up consists of lead vocalist Scott Stapp, guitarist and vocalist Mark Tremonti, bassist Brian Marshall, and drummer Scott Phillips.

 Creed released two studio albums, "My Own Prison" in 1997 and "Human Clay" in 1999, before Marshall left the band in 2000.

 The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between
members.

 Tremonti, Marshall, and Phillips went on to found Alter Bridge while Stapp followed a solo career.
</item 8>

<item 9>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 9>

<item 10>
Scott Weiland and the Wildabouts
Scott Weiland and the Wildabouts (sometimes billed simply as The Wildabouts) were a rock and roll band formed sometime around 2012 by former Stone Temple Pilots
and Velvet Revolver lead singer Scott Weiland, along with bassist Tommy Baker, guitarist Jeremy Brown, and multi-instrumentalist Doug Grean.

 The Wildabouts were marred with several roster changes (including the death of guitarist Jeremy Brown) and ultimately disbanded after Weiland's sudden death on
a tour bus in late 2015.
</item 10>
=== END ITEM id=73723b6b07f44b45 ===

=== ITEM id=e81a2ae4452e8eda ===
Gold sentences and facts:
Sentence 0: Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and Vietnam.
  Fact 0.0: Aspidistra is a genus of flowering plants.
  Fact 0.1: Aspidistra is in the family Asparagaceae.
  Fact 0.2: Aspidistra is in the subfamily Nolinoideae.
  Fact 0.3: Aspidistra is native to eastern Asia.
  Fact 0.4: Aspidistra is native to southeastern Asia.
  Fact 0.5: Aspidistra is particularly native to China.
  Fact 0.6: Aspidistra is particularly native to Vietnam.
Sentence 1: Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
  Fact 1.0: Cyrtanthus is a genus of perennial plants.
  Fact 1.1: Cyrtanthus is a genus of herbaceous plants.
  Fact 1.2: Cyrtanthus is a genus of bulbous plants.
  Fact 1.3: Cyrtanthus is in the family Amaryllidaceae.
  Fact 1.4: Cyrtanthus is in the subfamily Amaryllidoideae.

Retrieved items:
<item 1>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 1>

<item 2>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 2>

<item 3>
Ammocharis
Ammocharis is a small genus from sub-Saharan Africa, in the Amaryllidaceae family (subfamily Amaryllidoideae) which includes 7 species distributed in Africa.

 The plant grows as a succulent, above-ground bulb, preferring seasonally wet, hot, sandy soils and full sun.
</item 3>

<item 4>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 4>

<item 5>
Aspidistra
Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and
Vietnam.

 They grow in shade under trees and shrubs.

 Their leaves arise more or less directly from ground level, where their flowers also appear.

 The number of species known has increased considerably from the 1980s onwards, with around 100 accepted as of July 2013 . "

Aspidistra elatior" is common worldwide as a foliage house plant that is very tolerant of neglect.

 It and other species can also be grown in shade outside, where they are generally hardy to -5 C .
</item 5>

<item 6>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 6>

<item 7>
Amarylloidinae
Amarylloidinae is a now obsolete informal name for an "infrafamily" within the Amaryllidaceae (Amaryllis) family, erected by Hamilton Traub.

 This grouping was designed to fill a perceived gap between the formal rank of subfamily and tribes.

 In his treatment of this family, he divided it first into four subfamiles.

 Within subfamily Amarylloideae he then divided his sixteen tribes into two infrafamilies, Amarylloidinae (12 tribes) and Pancratioidinae (4 tribes), both of
which were subsequently demonstrated to be polyphyletic, and hence were abandoned by Dahlgren, who used no rank between family and tribe.

 On the other hand, he also used a much more restricted Amaryllidaceae corresponding to Traub's subfamily Amarylloideae.

 Thus Traub's Amarylloideae most closely resembles subfamily Amaryllidoideae "sensu" APGIII.
</item 7>

<item 8>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 8>

<item 9>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 9>

<item 10>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 10>
=== END ITEM id=e81a2ae4452e8eda ===

=== ITEM id=cf95874146d11ed6 ===
Gold sentences and facts:
Sentence 0: Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.
  Fact 0.0: Big Fish is a musical.
  Fact 0.1: Big Fish has music by Andrew Lippa.
  Fact 0.2: Big Fish has lyrics by Andrew Lippa.
  Fact 0.3: Big Fish has a book by John August.
Sentence 1:  He is a resident artist at the Ars Nova Theater in New York City.
  Fact 1.0: He is a resident artist at the Ars Nova Theater.
  Fact 1.1: The Ars Nova Theater is in New York City.

Retrieved items:
<item 1>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 1>

<item 2>
Andrew Lippa
Andrew Lippa (December 22, 1964) is an American composer, lyricist, book writer, performer, and producer.

 He is a resident artist at the Ars Nova Theater in New York City.
</item 2>

<item 3>
Marc Blitzstein
Blitzstein also composed music for films, such as "Surf and Seaweed" (1931) and "The Spanish Earth" (1937), and he contributed two songs to the original 1960
production of Hellman's play "Toys in the Attic".
</item 3>

<item 4>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 4>

<item 5>
Maurice Browne
Maurice Browne (12 February 1881 – 21 January 1955), born in Reading, England, was best known as a theater producer in the United States and the UK.

 The Cambridge-educated Browne was also a poet, actor, and theater director.

 He has been credited, along with his then-wife Ellen Van Volkenburg, with being the founder of the Little Theatre Movement in America through his work with the
Chicago Little Theatre.

 Browne and Van Volkenburg went on to found the department of drama at the Cornish School in Seattle in 1918, now Cornish College of the Arts.

 Browne's greatest triumph came in 1929 when he produced "Journey's End", by R. C. Sherriff in London.
</item 5>

<item 6>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 6>

<item 7>
Hamilton (album)
Hamilton is the cast album to the 2015 musical "Hamilton".

 The musical is based on the 2004 biography of Alexander Hamilton written by Ron Chernow, with music, lyrics, and book by Lin-Manuel Miranda.

 The recording stars Lin-Manuel Miranda, Leslie Odom Jr., Phillipa Soo, Renée Elise Goldsberry, Christopher Jackson, Daveed Diggs, Anthony Ramos, Okieriete
Onaodowan, Jasmine Cephas Jones, and Jonathan Groff.

 It also features Jon Rua, Thayne Jasperson, Sydney James Harcourt, Ariana DeBose, and Sasha Hutchings.

 It achieved the largest first week sales for a digital cast album and is the highest-charting cast album since 1963.

 It was the highest-selling Broadway cast album of 2015 and peaked at number one on the Rap Album chart, the first cast album to ever do so.
</item 7>

<item 8>
Love Birds (musical)
Love Birds is an original musical with book, music and lyrics by Robert J. Sherman.

 The musical officially premiered at the 2015 Edinburgh Fringe Festival at The Pleasance on August 7, 2015.

 The production was directed and choreographed by Stewart Nicholls.

 Prior to the Edinburgh run, there were two previews held in London at the Lost Theatre (in Stockwell) on July 28 and 29.

 The Original Edinburgh Cast Recording was recorded on July 30 and mixed and mastered on July 31 and August 3 and was first released by SimG Records on August
12 at the Edinburgh Festival.

 Two Edinburgh previews took place on August 5 and 6 with Press Night taking place on August 8.

 The final festival performance took place on August 31.

 The show starred Ruth Betteridge, Greg Castiglioni, John Guerrasio, George Knapper, Jonny Purchase, Joanna Sawyer, Anna Stolli, Rafe Watts and Ryan Willis.
</item 8>

<item 9>
Big Fish (musical)
Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.

 It is based on Daniel Wallace’s 1998 novel, "", and the 2003 film "Big Fish" written by August and directed by Tim Burton.
</item 9>

<item 10>
Robert Lopez
Robert Lopez (born February 23, 1975) is an American songwriter of musicals, best known for co-creating "The Book of Mormon" and "Avenue Q", and for composing
the songs featured in the Disney animated film "Frozen".

 He is the youngest of only twelve people who have won an Emmy, a Grammy, an Oscar and a Tony Award, and the quickest (10 years) to win all four.
</item 10>
=== END ITEM id=cf95874146d11ed6 ===

=== ITEM id=e5806fcd1b3248ad ===
Gold sentences and facts:
Sentence 0: The Monastery of the Mother of God at the Spring (full name in Greek: , pr.
  Fact 0.0: The full Greek name of the Monastery of the Mother of God at the Spring is given, with the pronunciation abbreviation 'pr.'
Sentence 1:  "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an Eastern Orthodox sanctuary in Istanbul, Turkey.
  Fact 1.0: The Monastery of the Mother of God at the Spring has the name 'Moni tis Theotóku tis Pigis'.
  Fact 1.1: The Monastery of the Mother of God at the Spring has the Turkish name 'Balıklı Meryem Ana Rum Manastiri'.
  Fact 1.2: The Monastery of the Mother of God at the Spring is simply called Zoödochos Pege.
  Fact 1.3: Zoödochos Pege is Greek for 'Life-giving Spring' (Ζωοδόχος Πηγή).
  Fact 1.4: The Monastery of the Mother of God at the Spring is an Eastern Orthodox sanctuary.
  Fact 1.5: The Monastery of the Mother of God at the Spring is located in Istanbul, Turkey.
Sentence 2: Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.
  Fact 2.0: Hagia Irene is also called Hagia Eirene.
  Fact 2.1: Hagia Irene has the Ancient Greek name Ἁγία Εἰρήνη.
  Fact 2.2: A Byzantine form of the name is given for Hagia Irene, but the text of that form is cut off in the source.
  Fact 2.3: Hagia Irene's name means 'Holy Peace'.
  Fact 2.4: Hagia Irene has the Turkish name 'Aya İrini'.
  Fact 2.5: Hagia Irene is sometimes known also as Saint Irene.
  Fact 2.6: Hagia Irene is a Greek Eastern Orthodox church.
  Fact 2.7: Hagia Irene is located in the outer courtyard of Topkapı Palace.
  Fact 2.8: Topkapı Palace is in Istanbul, Turkey.

Retrieved items:
<item 1>
Hagia Irene
Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek
Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.

 It is one of the few churches in Istanbul that has not been converted into a mosque.

 The Hagia Irene today operates as a museum and concert hall.
</item 1>

<item 2>
St. Mary's Church (Swormville, New York)
St. Mary's Catholic Church is a church located on the corner of Transit (New York State Route 78) and Stahley roads in Swormville, New York, in the United
States.

 Originally known as St. Mary’s Church of the Assumption at Transit, the building was constructed of about 260,000 bricks.

 John Nepomucene Neumann (later to be named St. John Neumann) started to visit Swormville, celebrating mass in local homes, barns and fields.

 In 1839 he instructed that a small house would be made to hold mass.

 Bishop John Timon established the first catholic church (St. Mary's) of the town of Clarence, New York.

 In 1861, Father Michael Schinabeak made plans for a church to be constructed.

 Construction began in 1862 and finished in fall of 1865.

 The official opening of the Church was in January 1866.

 In 2010 a new Church was constructed behind the old St. Mary's.
</item 2>

<item 3>
Churches and related institutions: Karlsøy’s 1854 wooden church seats 570 and holds occasional services; Manhattan’s Jesuit St. Ignatius parish began in 1851
and was renamed in 1898. Istanbul’s Spring monastery is an ancient Orthodox pilgrimage shrine with an 1835 church; Hagia Irene is an Orthodox museum and concert
hall. Ballston Spa’s St. Mary’s is the Albany Diocese’s fourth-oldest parish; Iowa City’s church and rect
</item 3>

<item 4>
Old Church of St George, Hill Street Dublin
The Old Church of St. George, commonly called "Little George's" in Hill Street (formerly Temple Street Lower) Parish of St. Mary, Dublin was built in 1668 by
the Eccles family for their workmen and also as a chapel-of-ease to the nearby St. Mary's Church.

 However, that St. Mary's Church was not St. Mary's Church, Dublin as its foundation stone was laid in 1700, and it was not St. Mary's Abbey, Dublin as it was
dissolved in 1539.

 Therefore St George's church, Hill Street, may have been a Chapel-of-Ease to St. Michan's Church in Church Street.

 The main body of the church, with the exception of the tower, was demolished in 1894.
</item 4>

<item 5>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 5>

<item 6>
Church of St. Mary of the Spring (Istanbul)
The Monastery of the Mother of God at the Spring (full name in Greek: , pr.

 "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an
Eastern Orthodox sanctuary in Istanbul, Turkey.

 The present church, built in 1835, bears the same dedication as the shrine erected in this place between the end of the fifth and the beginning of the sixth
century.

 After several renovations, this building was destroyed in the first half of the fifteenth century by the Ottomans.

 The complex got its name from a nearby holy spring, reputed to have healing properties.

 For almost fifteen hundred years, this sanctuary has been one of the most important pilgrimage sites of Greek Orthodoxy.
</item 6>

<item 7>
St Mary de Lode Church, Gloucester
St Mary de Lode Church, Archdeacon Street, Gloucester GL1 2QT, is a Church of England church immediately outside the grounds of Gloucester Cathedral.

 It is believed by some to be on the site of the first Christian church in Britain.

 The church is in the Diocese of Gloucester and Grade I listed by English Heritage.

 It has also been known as "St. Mary Before the Gate of St. Peter", "St. Mary Broad Gate" and "St. Mary De Port".
</item 7>

<item 8>
St. Mary's Episcopal Church (Washington, D.C.)
St. Mary's Episcopal Church, also known as St. Mary's, Foggy Bottom or St. Mary's Chapel, is a historic Episcopal church located at 730 23rd Street, N.W. in the
Foggy Bottom neighborhood of Washington, D.C..

 On April 2, 1973, St. Mary's Episcopal Church was added to the National Register of Historic Places.
</item 8>

<item 9>
St. Mary's Church and Rectory (Iowa City, Iowa)
St. Mary's Catholic Church, also known as St. Mary of the Visitation Church, is a parish church of the Diocese of Davenport which is located at 228 E. Jefferson
St. in Iowa City, Iowa, United States.

 The church building and rectory are listed together on the National Register of Historic Places in 1980.

 They were both included as contributing properties in the Jefferson Street Historic District in 2004.

 The parish's first rectory, which is now a private home, is also listed on the National Register as St. Mary's Rectory.

 It is located a few blocks to the east of the present church location at 610 E. Jefferson St.
</item 9>

<item 10>
St. Mary's Church (Ballston Spa, New York)
St. Mary's Church is a Catholic parish located in Ballston Spa, New York.

 It is located within the Roman Catholic Diocese of Albany.

 Father Thomas J. Kelly is the current pastor.

 St. Mary's is the fourth oldest parish in the Diocese.

 St. Mary's of Ballston Spa is partnered with St. Mary's of Galway, after the pastor at St. Mary's of Galway died and no replacement was available.
</item 10>
=== END ITEM id=e5806fcd1b3248ad ===

=== ITEM id=df840063d021a40c ===
Gold sentences and facts:
Sentence 0: Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.
  Fact 0.0: Charles Thomas 'Tom' Johnston was born on August 15, 1948.
  Fact 0.1: Charles Thomas 'Tom' Johnston is an American musician.
  Fact 0.2: Charles Thomas Johnston is known by the name 'Tom'.
Sentence 1: Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.
  Fact 1.0: Daryl Palumbo was born on February 10, 1979.
  Fact 1.1: Daryl Palumbo is an American musician.
  Fact 1.2: Daryl Palumbo is originally from Elmira, New York.

Retrieved items:
<item 1>
Head Automatica
Head Automatica is an American rock band from Long Island, New York fronted by Daryl Palumbo (also of Glassjaw).
</item 1>

<item 2>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 2>

<item 3>
Mike Byrne (musician)
Michael William Byrne (born February 6, 1990) is a drummer who was a member of the band The Smashing Pumpkins.

 When Pumpkins frontman Billy Corgan called for rehearsals to replace Jimmy Chamberlin, Byrne auditioned and was chosen out of thousands of applicants.

 He played drums in Sky Saxon tribute band Spirits in the Sky for 6 shows in August 2009, along with Corgan, Kerry Brown, Kevin Dippold, Mark Tulin, Linda
Strawberry, Ysanne Spevack, Mark Weitz, and Dave Navarro, and also performs drums on the ongoing Pumpkins 44-song project, "Teargarden by Kaleidyscope".

 As a part of "Teargarden", Mike contributed drumming duties on "Oceania", the Pumpkins' eighth full-length album.

 On April 17, 2010, Byrne played his first show under the Smashing Pumpkins moniker in celebration of Record Store Day.
</item 3>

<item 4>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 4>

<item 5>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 5>

<item 6>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 6>

<item 7>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 7>

<item 8>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 8>

<item 9>
Tom Johnston
Tom Johnston, Thomas Johnston or Tommy Johnston may refer to:
</item 9>

<item 10>
Daryl Palumbo
Daryl Palumbo (born February 10, 1979) is an American musician, originally from Elmira, New York.

 He is the frontman of the bands Glassjaw, Head Automatica and Color Film.

 As a youth he was a member of the Long Island straight edge band "XbustedX."

 In 1993 he met guitarist Justin Beck, who would become his good friend.

 Together they formed Glassjaw.
</item 10>
=== END ITEM id=df840063d021a40c ===
