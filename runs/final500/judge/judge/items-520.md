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

=== ITEM id=283680617ab38cde ===
Gold sentences and facts:
Sentence 0: Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.
  Fact 0.0: Catfish and the Bottlemen are a British rock band.
  Fact 0.1: Catfish and the Bottlemen were formed in Llandudno, Wales.
  Fact 0.2: Catfish and the Bottlemen were formed in 2007.
Sentence 1: The Lumineers are an American folk rock/Americana band based in Denver, Colorado.
  Fact 1.0: The Lumineers are an American folk rock/Americana band.
  Fact 1.1: The Lumineers are based in Denver, Colorado.
Sentence 2:  Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.
  Fact 2.0: Schultz and Fraites began writing together in 2005.
  Fact 2.1: Schultz and Fraites began performing together in 2005.
  Fact 2.2: Schultz and Fraites began writing together in Ramsey, New Jersey.
  Fact 2.3: Schultz and Fraites began performing together in Ramsey, New Jersey.
Sentence 3:  The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.
  Fact 3.0: The Lumineers emerged as one of the most popular folk rock/Americana artists.
  Fact 3.1: The Lumineers emerged during the revival of folk rock and Americana genres.
  Fact 3.2: The Lumineers emerged during the growing popularity of folk rock and Americana in the 2010s.

Retrieved items:
<item 1>
The Ride Tour
The Ride Tour is the ongoing second concert tour by English indie rock band Catfish and the Bottlemen, in support of the group's second studio album "The Ride".

 The tour was announced in February 2016 and began on 4 April 2016, in Glasgow, United Kingdom at the O2 Academy Glasgow.

 The tour is planned to travel across Europe and North America with a total of 50 shows planned so far.

 On 22 February 2016, the band announced an April 2016 stint of shows in UK, including Glasgow, Halifax, Southend On Sea, Doncaster, London and Brighton.

 The band will also headline several festivals across the UK throughout the summer, including Liverpool's 'Sound City' in May 2016, Bristol's Summer Series in
July 2016 and Manchester's Castlebowl in July 2016.

 On 14 March 2016 it was announced that the band will play several dates in North America in June 2016.
</item 1>

<item 2>
The Balcony (album)
The Balcony is the debut studio album by Welsh rock band Catfish and the Bottlemen.

 It was released on 15 September 2014 in the United Kingdom and 15 January 2015 in the United States.

 On 20 March 2015, "The Balcony" was awarded a Gold certification from the British Phonographic Industry.

 The album cover depicts outlines of two headless people pleasuring each other's genitals.

 The illustration is by New York artist Tim Lahan who originally posted it to his Flickr account in 2009 and was subsequently contacted by the band.
</item 2>

<item 3>
Poco
Poco is an American country rock band originally formed by Richie Furay, Jim Messina and Rusty Young.

 Formed following the demise of Buffalo Springfield in 1968, Poco was part of the first wave of the West Coast country rock genre.

 The title of their first album, "Pickin' Up the Pieces", is a reference to the break-up of Buffalo Springfield.

 Throughout the years Poco has performed in various groupings, and is still active.
</item 3>

<item 4>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 4>

<item 5>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 5>

<item 6>
Beautiful Decay
Beautiful Decay is the second extended play by Welsh indie rock band, Catfish and the Bottlemen.

 Released on 10 March 2010, the EP contained one song that would later be featured in their debut studio album, "The Balcony".
</item 6>

<item 7>
The Ride (Catfish and the Bottlemen album)
The Ride is the second studio album by British indie rock band Catfish and the Bottlemen.

 It was released on 27 May 2016 through Capitol Records.
</item 7>

<item 8>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 8>

<item 9>
Kathleen (song)
"Kathleen" is the third single by Welsh indie rock band, Catfish and the Bottlemen.

 The song was included in their EP, "Kathleen and the Other Three", and their debut studio album, "The Balcony".

 The single was digitally released on 18 March 2014 as a music video, and on 7 April 2014, the 7-inch version of the single was released.

 The single did not contain a B-side.
</item 9>

<item 10>
Billy Bibby &amp; The Wry Smiles
Billy Bibby & The Wry Smiles are a British indie rock band formed in the autumn of 2015 in the area around Llandudno, North Wales and Chester, UK.

 The band is fronted by Billy Bibby, former lead guitarist and founding member of Catfish and the Bottlemen.

 Bibby parted ways with Catfish and the Bottlemen in the summer of 2014, shortly after co-writing and recording their Platinum selling debut album, "The
Balcony".

 Soon thereafter, Billy Bibby began writing songs and playing acoustic shows in various UK cities in order to test public response to his new music.

 In the spring of 2015, Billy Bibby began seeking musicians to form a new band.

 Auditions were conducted with the help of Simon Jones, formerly of The Verve.

 Band members Rob Jones (lead guitar), Matt Bassy (bass), and Michael Pearce (drums) were selected to form the four-piece ensemble.
</item 10>
=== END ITEM id=283680617ab38cde ===

=== ITEM id=52114d1f3a01956e ===
Gold sentences and facts:
Sentence 0: Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).
  Fact 0.0: Dictyosperma is a monotypic genus of flowering plant.
  Fact 0.1: Dictyosperma is in the palm family.
  Fact 0.2: Dictyosperma is found in the Mascarene Islands in the Indian Ocean.
  Fact 0.3: Dictyosperma is found in Mauritius.
  Fact 0.4: Dictyosperma is found in Réunion.
  Fact 0.5: Dictyosperma is found in Rodrigues.
Sentence 1: The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in 1810.
  Fact 1.0: The genus Huernia is in the family Apocynaceae.
  Fact 1.1: The genus Huernia is in the subfamily Asclepiadoideae.
  Fact 1.2: The genus Huernia consists of stem succulents.
  Fact 1.3: The stem succulents of the genus Huernia are from Eastern Africa.
  Fact 1.4: The stem succulents of the genus Huernia are from Southern Africa.
  Fact 1.5: The genus Huernia was first described as a genus in 1810.

Retrieved items:
<item 1>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 1>

<item 2>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 2>

<item 3>
Harrimanella
Harrimanella is a genus of flowering plant in the heath family Ericaceae, with a single species, Harrimanella hypnoides, also known as moss bell heather.

 It was originally named Cassiope hypnoides by Linnaeus (1737) in his Flora Lapponica, but "Harrimanella hypnoides" is now the accepted name at ITIS.

 The species name "hypnoides" means 'like "Hypnum" ', which is a genus mosses.
</item 3>

<item 4>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 4>

<item 5>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 5>

<item 6>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 6>

<item 7>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 7>

<item 8>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 8>

<item 9>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 9>

<item 10>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 10>
=== END ITEM id=52114d1f3a01956e ===

=== ITEM id=9c60db907d065e4e ===
Gold sentences and facts:
Sentence 0: The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the headquarters of book publisher Random House and a luxury apartment complex.
  Fact 0.0: The Random House Tower is also known as the Park Imperial Apartments.
  Fact 0.1: The Random House Tower is a 52-story mixed-use tower.
  Fact 0.2: The Random House Tower is in New York City, United States.
  Fact 0.3: The Random House Tower is used as the headquarters of book publisher Random House.
  Fact 0.4: The Random House Tower is used as a luxury apartment complex.
Sentence 1:  It currently carries the Vornado Realty Trust corporate headquarters.
  Fact 1.0: The Random House Tower currently carries the Vornado Realty Trust corporate headquarters.

Retrieved items:
<item 1>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 1>

<item 2>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 2>

<item 3>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 3>

<item 4>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
</item 4>

<item 5>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 5>

<item 6>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
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
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
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
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 10>
=== END ITEM id=9c60db907d065e4e ===

=== ITEM id=5a1276ce66170a03 ===
Gold sentences and facts:
Sentence 0: Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.
  Fact 0.0: Triumph of the Will's German title is Triumph des Willens.
  Fact 0.1: Triumph of the Will is a 1935 German propaganda film.
  Fact 0.2: Leni Riefenstahl directed Triumph of the Will.
  Fact 0.3: Leni Riefenstahl produced Triumph of the Will.
  Fact 0.4: Leni Riefenstahl edited Triumph of the Will.
  Fact 0.5: Leni Riefenstahl co-wrote Triumph of the Will.
Sentence 1: The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".
  Fact 1.0: The End of America is a 2008 documentary film.
  Fact 1.1: Annie Sundberg directed The End of America.
  Fact 1.2: Ricki Stern directed The End of America.
  Fact 1.3: The End of America is adapted from Naomi Wolf's 2007 book.

Retrieved items:
<item 1>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 1>

<item 2>
The End of America (film)
The End of America is a 2008 documentary film directed by Annie Sundberg and Ricki Stern, adapted from Naomi Wolf's 2007 book, "".

 The film analyzes policy changes made during the Bush Administration, and makes the case that these changes threaten American democracy.
</item 2>

<item 3>
The Year of the Voyager
The Year Of The Voyager is a double DVD/CD released by Nevermore.

 It was released in Europe on October 20, 2008 and in North America on November 25, 2008 via Century Media.

 The set covers the "This Godless Endeavor" touring cycle, starting with live footage from the U.S. Gigantour 2005, the Metal Mania festival 2006 in Poland, the
Wacken Open Air festival in 2006 in Germany and the main DVD show recorded at the Zeche in Bochum, Germany.

 Bonus material includes two songs from Century Media USA 10th Anniversary Party 2001, all promo videos and an interview with singer Warrel Dane which was
recorded at the Roax Film Studios in Berlin in the spring of 2008.
</item 3>

<item 4>
Apocalypse Now
Apocalypse Now is a 1979 American epic war film directed, produced, and co-written by Francis Ford Coppola.

 It was co-written by John Milius with narration written by Michael Herr.

 It stars Marlon Brando, Robert Duvall, Martin Sheen, Frederic Forrest, Albert Hall, Sam Bottoms, Larry Fishburne, and Dennis Hopper.

 The screenplay, written by Milius, adapts the story of Joseph Conrad's novella "Heart of Darkness", changing its setting from late nineteenth-century Congo to
the Vietnam War.

 It draws from Herr's "Dispatches" and Werner Herzog's "Aguirre, the Wrath of God" (1972).

 The film revolves around Captain Benjamin L. Willard (Sheen) on a secret mission to assassinate Colonel Kurtz, a renegade who is presumed insane.
</item 4>

<item 5>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 5>

<item 6>
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
</item 6>

<item 7>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 7>

<item 8>
Love Begins at 20
Love Begins at 20 is a 1936 American comedy film directed by Frank McDonald and written by Dalton Trumbo and Tom Reed, based on the 1929 play "Broken Dishes" by
Martin Flavin.

 The film stars Hugh Herbert, Patricia Ellis, Warren Hull, Hobart Cavanaugh, Dorothy Vaughan and Clarence Wilson.

 The film was released by Warner Bros. on August 22, 1936.
</item 8>

<item 9>
Empire (1964 film)
Empire is a 1964 American black and white silent film written, produced, and directed by Andy Warhol.

 It consists of eight hours and five minutes of continuous slow motion footage of the Empire State Building in New York City.

 Abridged showings of the film were never allowed, and supposedly the unwatchability of the film was an important part of the reason the film was created.

 However, a legitimate Italian VHS produced in association with the Andy Warhol Museum in 2000 contains only an extract of 60 minutes.

 Its use of the long take "in extremis" is an extension of Warhol's earlier work the previous year with "Sleep".

 Warhol employed Rob Trains to be the projectionist for a screening of the film.

 Trains miscalculated and mixed the order and speed of the reels for the eight-hour movie.

 After a positive review in "The New York Times", Warhol actually liked the "mistake" and employed Trains for the entire summer.
</item 9>

<item 10>
Triumph of the Will
Triumph of the Will (German: Triumph des Willens ) is a 1935 German propaganda film directed, produced, edited, and co-written by Leni Riefenstahl.

 It chronicles the 1934 Nazi Party Congress in Nuremberg, which was attended by more than 700,000 Nazi supporters.

 The film contains excerpts from speeches given by Nazi leaders at the Congress, including Adolf Hitler, Rudolf Hess and Julius Streicher, interspersed with
footage of massed Sturmabteilung (SA) and Schutzstaffel (SS) troops and public reaction.

 Hitler commissioned the film and served as an unofficial executive producer; his name appears in the opening titles.

 The film's overriding theme is the return of Germany as a great power, with Hitler as the leader who will bring glory to the nation.

 Because the film was made after the 1934 Night of the Long Knives (on 30 June) many prominent Sturmabteilung (SA) members are absent—they were murdered in that
Party purge, organised and orchestrated by Hitler to replace the SA with the Schutzstaffel (SS) as his main paramilitary force.
</item 10>
=== END ITEM id=5a1276ce66170a03 ===

=== ITEM id=738f1ddc39001e5f ===
Gold sentences and facts:
Sentence 0: Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant", his ten-novel fantasy series.
  Fact 0.0: Stephen Reeder Donaldson was born on May 13, 1947.
  Fact 0.1: Stephen Reeder Donaldson is an American novelist.
  Fact 0.2: Stephen Reeder Donaldson writes fantasy novels.
  Fact 0.3: Stephen Reeder Donaldson writes science fiction novels.
  Fact 0.4: Stephen Reeder Donaldson writes mystery novels.
  Fact 0.5: Stephen Reeder Donaldson is most famous for "The Chronicles of Thomas Covenant".
  Fact 0.6: "The Chronicles of Thomas Covenant" is a ten-novel fantasy series.
  Fact 0.7: "The Chronicles of Thomas Covenant" is a series by Stephen Reeder Donaldson.
Sentence 1: Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.
  Fact 1.0: Michael John Moorcock was born on 18 December 1939.
  Fact 1.1: Michael John Moorcock is an English writer.
  Fact 1.2: Michael John Moorcock is primarily a writer of science fiction.
  Fact 1.3: Michael John Moorcock is primarily a writer of fantasy.
  Fact 1.4: Michael John Moorcock has also published literary novels.

Retrieved items:
<item 1>
The Final Programme
The Final Programme is a novel by British science fiction and fantasy writer Michael Moorcock.

 Written in 1965 as the underground culture was beginning to emerge, it was not published for several years.

 Moorcock has stated that publishers at the time considered it was "too freaky".
</item 1>

<item 2>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 2>

<item 3>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 3>

<item 4>
Behold the Man (novel)
Behold the Man (1969) is a science fiction novel by British writer Michael Moorcock.

 It originally appeared as a novella in a 1966 issue of "New Worlds"; later, Moorcock produced an expanded version which was first published in 1969 by Allison
& Busby.

 The title derives from the Gospel of John, Chapter 19, Verse 5: "Then Jesus came out, wearing the crown of thorns and the purple robe.

 And Pilate said to them Behold the Man."
</item 4>

<item 5>
Black Blade (song)
"Black Blade" is a song by American rock band Blue Öyster Cult.

 The lyrics were written by Eric Bloom and British science fiction writer Michael Moorcock from the viewpoint of Elric, Moorcock's most famous character.

 Elric wields a sentient, soul-sucking sword named Stormbringer.

 Aside from a set of cryptic, blood-red runes on the blade, the sword is black, hence the name "Black Blade".

 The music was jointly composed by Eric Bloom, lead singer of Blue Öyster Cult, and bass-player John Trivers.

 The song was featured in the Blue Öyster Cult album "Cultösaurus Erectus."

 When Bloom was asked how he got in touch with Moorcock for the song he claims, "I went out of my way to send him a Fan Boy letter.

 He was living in England, and he came over to America, to meet with his publisher.

 He said, “Let's get together.”

 We got together, and bonded, and he started sending me lyrics.

 That is how "Black Blade," "Veteran of Psychic Wars" and "Great Sun Jester" happened.
</item 5>

<item 6>
Douglas Hill
Douglas Arthur Hill (6 April 1935 – 21 June 2007) was a Canadian science fiction author, editor and reviewer.

 He was born in Brandon, Manitoba, the son of a railroad engineer, and was raised in Prince Albert, Saskatchewan.

 An avid science fiction reader from an early age, he studied English at the University of Saskatchewan (where he earned an Honours B.A. in 1957) and at the
University of Toronto.

 He married fellow writer and U. of S. alumna Gail Robinson (writer) in 1958; they moved to Britain in 1959, where he worked as a freelance writer and editor
for Aldus Books.

 In 1967–1968 he served as Assistant Editor of the controversial "New Worlds" science fiction magazine under Michael Moorcock.
</item 6>

<item 7>
The Chinese Agent
The Chinese Agent (1970) is a comic novel by Michael Moorcock.

 It is a revision of "Somewhere in the Night", which Moorcock published in 1966 under the pseudonym Bill Barclay.

 Although Moorcock is best known as the author of fantasy fiction and science fiction-based parables such as "Behold the Man" and "The Dancers at the End of
Time", here he writes a light-hearted caper that parodies the spy novel genre.
</item 7>

<item 8>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 8>

<item 9>
A Cure for Cancer
A Cure for Cancer is a novel by British fantasy and science fiction writer Michael Moorcock, first published in London 1971 by Allison and Busby.

 The book is part of Moorcock's long-running Jerry Cornelius series.
</item 9>

<item 10>
Michael Moorcock
Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.

 He is best known for his novels about the character Elric of Melniboné, a seminal influence on the field of fantasy in the 1960s and 1970s.
</item 10>
=== END ITEM id=738f1ddc39001e5f ===

=== ITEM id=1fff541da3bf716d ===
Gold sentences and facts:
Sentence 0: Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11, 1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical physics.
  Fact 0.0: Nikolay Mitrofanovich Krylov was born on 29 November [O.S. 17 November] 1879.
  Fact 0.1: Nikolay Mitrofanovich Krylov died on May 11, 1955.
  Fact 0.2: Nikolay Mitrofanovich Krylov's Russian name is Николай Митрофанович Крылов.
  Fact 0.3: Nikolay Mitrofanovich Krylov's Ukrainian name is Микола Митрофанович Крилов.
  Fact 0.4: Nikolay Mitrofanovich Krylov was a Russian mathematician.
  Fact 0.5: Nikolay Mitrofanovich Krylov was a Soviet mathematician.
  Fact 0.6: Nikolay Mitrofanovich Krylov is known for works on interpolation.
  Fact 0.7: Nikolay Mitrofanovich Krylov is known for works on non-linear mechanics.
  Fact 0.8: Nikolay Mitrofanovich Krylov is known for works on numerical methods for solving equations of mathematical physics.
Sentence 1: Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.
  Fact 1.0: Anatoly Timofeevich Fomenko's Russian name is Анатолий Тимофеевич Фоменко.
  Fact 1.1: Anatoly Timofeevich Fomenko was born on 13 March 1945 in Stalino, USSR.
  Fact 1.2: Anatoly Timofeevich Fomenko is a Soviet mathematician.
  Fact 1.3: Anatoly Timofeevich Fomenko is a Russian mathematician.
  Fact 1.4: Anatoly Timofeevich Fomenko is a professor at Moscow State University.
  Fact 1.5: Anatoly Timofeevich Fomenko is well known as a topologist.
  Fact 1.6: Anatoly Timofeevich Fomenko is a member of the Russian Academy of Sciences.

Retrieved items:
<item 1>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 1>

<item 2>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 2>

<item 3>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 3>

<item 4>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 4>

<item 5>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 5>

<item 6>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 6>

<item 7>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 7>

<item 8>
Vladimir Semenikhin
Vladimir Anatolievich Semenikhin (born August 31, 1967) is a Russian businessman, philanthropist, art collector and honorary consul.

 Semenikhin is the president of the Ekaterina Cultural Foundation, Chairman of the board of directors at Stroyteks Group of companies and the Honorary Consul of
Kazakhstan in the Principality of Monaco.
</item 8>

<item 9>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 9>

<item 10>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 10>
=== END ITEM id=1fff541da3bf716d ===

=== ITEM id=7b12b9e977ea006a ===
Gold sentences and facts:
Sentence 0:  His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".
  Fact 0.0: The exploits of the man referred to here in that battle were described by Hal Moore.
  Fact 0.1: Hal Moore described the exploits in the book 'We Were Soldiers Once and Young'.
Sentence 1: Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.
  Fact 1.0: Harold Gregory 'Hal' Moore, Jr. was born on February 13, 1922.
  Fact 1.1: Harold Gregory 'Hal' Moore, Jr. died on February 10, 2017.
  Fact 1.2: Harold Gregory 'Hal' Moore, Jr. was a United States Army lieutenant general.
  Fact 1.3: Harold Gregory 'Hal' Moore, Jr. was an author.

Retrieved items:
<item 1>
Motor Boys
The Motor Boys were the heroes of a popular series of adventure books for boys at the turn of the 20th century issued by the Stratemeyer Syndicate under the
pseudonym of Clarence Young.

 This series was published by Cupples & Leon and was issued with dustjackets and glossy frontispiece.

 Howard Garis (author of the Uncle Wiggily stories) wrote many, if not all, of these stories.
</item 1>

<item 2>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 2>

<item 3>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 3>

<item 4>
Graph query row:
person: Hal Moore
military_rank: ['United States Army lieutenant general', 'author']
relationship: CREATED_BY
connected_to: We Were Soldiers Once and Young
</item 4>

<item 5>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 5>

<item 6>
Myron F. Diduryk
Myron F. Diduryk (July 15, 1938 to April 24, 1970) was a Ukrainian-American United States Army Major, who played a key role as an infantry company commander in
the Battle of Ia Drang, the first major battle of the Vietnam War.

 His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".

 Moore said that Diduryk was, “… the finest battlefield company commander I had ever seen, bar none.”

 Diduryk was killed in action on his second tour in Vietnam.
</item 6>

<item 7>
William K. MacNulty
William Kirk MacNulty (May 22, 1892 - August 3, 1964) was a U. S. Marine with a long and distinguished record.

 He was a young U.S. Marine Corps Second Lieutenant during World War One and saw action at the Battle of the Argonne Forest.

 He served as a captain during the Second U.S. Nicaraguan Campaign (1926–33).

 During the Second World War as Lieutenant Colonel he commanded the U.S. Marine Corps defense of Guam against overwhelming Imperial Japanese forces during the
First Battle of Guam.

 He was incarcerated by the Japanese as a prisoners of war.

 He was promoted to Brigadier General during captivity and retired from military service in 1946.

 He is buried at the Golden Gate National Cemetery in San Bruno, San Mateo County, California.
</item 7>

<item 8>
Dispatches (book)
Dispatches is a New Journalism book by Michael Herr that describes the author's experiences in Vietnam as a war correspondent for "Esquire" magazine.

 First published in 1977, "Dispatches" was one of the first pieces of American literature that portrayed the experiences of soldiers in the Vietnam War for
American readers.
</item 8>

<item 9>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 9>

<item 10>
The Great War for Civilisation
The Great War for Civilisation: The Conquest of the Middle East is a book published in 2005 by the award-winning English journalist Robert Fisk.

 The book is a compilation of many of the articles Fisk wrote when he was serving as a correspondent in the Middle East for "The Times" and "The Independent".

 The book revolves around several key themes regarding the history of the modern Middle East: the Arab-Israeli conflict, the Soviet invasion of Afghanistan, the
Persian Gulf War as well as the 2003 Iraq War (United States invasion of Iraq) as well as other regional conflicts such as the Armenian Genocide and the
Algerian Civil War.

 The "Great War for Civilisation" is the second book Fisk has written about the Middle East with the first one, "Pity the Nation", (Nation Books, 2002) being
about the Lebanese Civil War.
</item 10>
=== END ITEM id=7b12b9e977ea006a ===

=== ITEM id=4a02e0342ee0618f ===
Gold sentences and facts:
Sentence 0:  The trains were built between 1994 and 2009 – for the most part by Talbot, part of Bombardier Inc., with some railcars built by De Dietrich.
  Fact 0.0: The trains were built between 1994 and 2009.
  Fact 0.1: The trains were built for the most part by Talbot.
  Fact 0.2: Talbot is part of Bombardier Inc.
  Fact 0.3: Some railcars were built by De Dietrich.
Sentence 1: Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.
  Fact 1.0: Bombardier Inc. is a multinational aerospace and transportation company.
  Fact 1.1: Bombardier Inc. is based in Montreal, Quebec, Canada.

Retrieved items:
<item 1>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 1>

<item 2>
Marquette Transportation Company
The Marquette Transportation Company is a marine transportation company based in Paducah, Kentucky, United States.
</item 2>

<item 3>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
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
New York Transportation Company
The New York Transportation Company (originally New York Electrical Vehicle Transportation Company) was a company incorporated in New Jersey in 1899 as the 'New
York Electrical Vehicle Transportation Company' which changed its name to the "New York Transportation Company" in 1902 and was wound up in 1936.

 From 1922 it was controlled by the 'Fifth Avenue Bus Securities Corporation' which was incorporated in Delaware in 1922 which was controlled by The Omnibus
Corporation also incorporated in Delaware.
</item 5>

<item 6>
Toland, Pennsylvania
Toland is less than 1/4 of a mile long, and it is located less than half a mile from the Appalachian trail crossing.

 The closest town is Mount Holly Springs, Pennsylvania located (3 miles north of Toland).

 The village is roughly 10 miles south of Carlisle, Pennsylvania, 7 miles south east of Boiling Springs, Pennsylvania, and 15 miles South of Gettysburg,
Pennsylvania.

 The only business in Toland is the Cherokee Campground formerly known as the Tagg Run Campground.

 The campground restaurant is now closed.

 Toland Mission is a small non-denominational church that can hold up to 74 persons.

 It was originally built by the owner of Beetem Lumber Company in Carlisle for the families of the community of Toland.

 A one-room school house near the church originally served the community, but it was converted to a home when residents' children were transported to a township
school.

 Local people in Toland usually travel to the town of Mount Holly Springs for necessities such as gas, food, and toiletries.
</item 6>

<item 7>
Merchants Transportation Company
The Merchants Transportation Company was a shipping firm that operated on Puget Sound from 1905 to 1929.

 This company should not be confused with the similarly named Merchants Transportation Company of Olympia, formed in 1874 and a completely separate firm.
</item 7>

<item 8>
Transportation in Jacksonville, Florida
The Jacksonville transportation network includes ground, air, and sea options for passenger and freight transit.

 The Jacksonville Port Authority (Jaxport) operates the Port of Jacksonville, which includes container shipping facilities at "Blount Island Marine Terminal",
the "Talleyrand Marine Terminal" and the "Dames Point Marine Terminal".

 Jacksonville Aviation Authority managers Jacksonville International Airport in Northside, as well as several smaller airports.

 The Jacksonville Transportation Authority (JTA) operates bus, people mover, and park-n-ride services throughout the city and region.

 A major bus terminal at the intermodal Rosa Parks Transit Station serves as JTA's main transit hub.

 Various intercity bus companies terminate near Central Station.

 Amtrak operates passenger rail service to and from major cities throughout North America.

 The city is bisected by major highways, I-95 and I-10, I-295 creates a full beltway around the city.
</item 8>

<item 9>
Toland, Pennsylvania
Toland is an unincorporated community in Cumberland County, Pennsylvania, United States.

 The very small village is located on Pine Grove Road, east of Mountain Creek Campground.

 Michaux State Forest, the site of Laurel Lake, Fuller Lake and Pine Grove Furnace State Park lie a few miles to the west near the intersection of Pine Grove
Road and PA route 233.

 Toland has a mailing address of Gardners Pennsylvania, because the size of the community doesn't warrant a post-office, nor an official incorporated name.

 What is now a sand pit operation was originally a clay bank mining operation that was the reason for the location of the village of Toland.

 Less than 50 people reside in the close-knit community, which was built for the clay bank company workers in the first quarter of the 20th century.

 The original community of Toland, Pennsylvania consisted of 11 duplex houses, built side by side along Pine Grove Road, with less than 0.17 of an acre of land
to each.

 Additional homes have been built since.

 When the community was originally built there was only a common well with a hand pump for all of the families to draw from.
</item 9>

<item 10>
Northern Transportation Company
Northern Transportation Company Limited (NTCL) is a marine transportation company in the Canadian and American Arctic owned by Norterra, a holding company that
is owned by the Inuvialuit of the Northwest Territories.

 While primarily a marine freight hauler on Canadian rivers, like the Mackenzie and the Hay and along the Arctic coast of Canada, they are also a petroleum
wholesaler.

 Its head office is now in the town of Hay River, Northwest Territories.

 Today, it primarily utilizes tugs and barges.
</item 10>
=== END ITEM id=4a02e0342ee0618f ===

=== ITEM id=f8f8996d5162ef05 ===
Gold sentences and facts:
Sentence 0: Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
  Fact 0.0: Giuseppe Fortunino Francesco Verdi was born on 9 or 10 October 1813.
  Fact 0.1: Giuseppe Fortunino Francesco Verdi died on 27 January 1901.
  Fact 0.2: Giuseppe Fortunino Francesco Verdi was an Italian opera composer.
Sentence 1: Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
  Fact 1.0: Charles Louis Ambroise Thomas was born on 5 August 1811.
  Fact 1.1: Charles Louis Ambroise Thomas died on 12 February 1896.
  Fact 1.2: Charles Louis Ambroise Thomas was a French composer.
  Fact 1.3: Charles Louis Ambroise Thomas is best known for his opera 'Mignon' (1866).
  Fact 1.4: Charles Louis Ambroise Thomas is best known for his opera 'Hamlet' (1868).
  Fact 1.5: The opera 'Hamlet' (1868) is after Shakespeare.
  Fact 1.6: Charles Louis Ambroise Thomas is best known as Director of the Conservatoire de Paris.
  Fact 1.7: Charles Louis Ambroise Thomas was Director of the Conservatoire de Paris from 1871 till his death.

Retrieved items:
<item 1>
Giuseppe Verdi (film)
Giuseppe Verdi is a 1938 Italian biographical film directed by Carmine Gallone and starring Fosco Giachetti, Gaby Morlay and Germana Paolieri.

 The film portrays the life of the composer Giuseppe Verdi (1813-1901).

 The casting of Giachetti as Verdi was intended to emphasise the composer's patriotism, as he had recently played patriotic roles in films such as "The White
Squadron".

 The film was made at the Cinecittà Studios in Rome.

 The film is also known by the alternative title The Life of Giuseppe Verdi.
</item 1>

<item 2>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 2>

<item 3>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 3>

<item 4>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 4>

<item 5>
Ambroise Thomas
Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after
Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.
</item 5>

<item 6>
Ferdinando Provesi
Ferdinando Angelo Maria Provesi (1770 – 1833) was a native of Parma, Italy.

 He was regarded as one of the greatest Italian opera composers of the era .

 Provesi is best known as being an early tutor of Giuseppe Verdi when he was the "Maestro di cappella" (master of music) at the St. Bartolomeo cathedral in
Busseto (the town very close to Le Roncole, the village where Verdi was born.)

 Provesi was also director of the municipal music school and local Philharmonic Society.

 He began teaching Verdi in 1824 when the future composer was 11 years old.
</item 6>

<item 7>
Teatro Giuseppe Verdi
Teatro Giuseppe Verdi (the Giuseppe Verdi Theatre) is a small opera house located in a wing of the Rocca dei Marchesi Pallavicino on the Piazza Giuseppe Verdi
in Busseto, Italy, a town closely associated with the life of the opera composer, Giuseppe Verdi.

 From the 13th century, the “rocca” or “fortress” was the family’s palace; it is now the city hall after being acquired by the municipality in 1856.

 The theatre opened on 15 August 1868 and seats 300.
</item 7>

<item 8>
Alessandro Scarlatti
Pietro Alessandro Gaspare Scarlatti (2 May 1660 – 22 October 1725) was an Italian Baroque composer, especially famous for his operas and chamber cantatas.

 He is considered the founder of the Neapolitan school of opera.

 He was the father of two other composers, Domenico Scarlatti and Pietro Filippo Scarlatti.
</item 8>

<item 9>
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 9>

<item 10>
Verdi, the King of Melody
Giuseppe Verdi, released theatrically in the USA as The Life and Music of Giuseppe Verdi and on video as Verdi, the King of Melody, is a 1953 Italian
biographical film starring Pierre Cressoy and directed by Raffaello Matarazzo.

 It is based on adult life events of the composer Giuseppe Verdi.

 The film was a commercial success, grossing over 957 million lire at the Italian box office.
</item 10>
=== END ITEM id=f8f8996d5162ef05 ===

=== ITEM id=6079850329ff49a3 ===
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
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
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
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 3>

<item 4>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 4>

<item 5>
TriBond
TriBond is a board game that has sold over 3 million copies in 14 countries since its release in 1990.

 It requires players to determine a common bond between three subjects.

 It follows in the tradition of "Trivial Pursuit", "Outburst" and other adult boardgames that require a wide range of knowledge but "TriBond" requires some
problem solving ability as well.
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
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 9>

<item 10>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 10>
=== END ITEM id=6079850329ff49a3 ===

=== ITEM id=74a2178359ddbff3 ===
Gold sentences and facts:
Sentence 0:  Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.
  Fact 0.0: Players compete in races.
  Fact 0.1: The race tracks are inspired by attractions such as the Haunted Mansion.
  Fact 0.2: The race tracks are inspired by attractions such as Pirates of the Caribbean.
  Fact 0.3: The race tracks are inspired by attractions such as Space Mountain.
  Fact 0.4: The race tracks are inspired by attractions such as Big Thunder Mountain Railroad.
  Fact 0.5: Players race on tracks to acquire missing parts for the park's fireworks machine.
  Fact 0.6: The park's fireworks machine was accidentally destroyed by Chip 'n' Dale.
  Fact 0.7: Chip 'n' Dale destroyed the fireworks machine while they were gathering acorns.
Sentence 1:  Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.
  Fact 1.0: Space Mountain opened on January 15, 1975.
  Fact 1.1: Space Mountain is the oldest operating roller coaster in the state of Florida.
  Fact 1.2: Space Mountain is the original version of an iconic attraction.
  Fact 1.3: The original version of Space Mountain has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide.
  Fact 1.4: The replication of Space Mountain excludes the Shanghai Disneyland Resort.

Retrieved items:
<item 1>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 1>

<item 2>
Millennium Force
Millennium Force is a steel roller coaster built by Intamin at Cedar Point amusement park in Sandusky, Ohio, United States.

 It was the fourteenth roller coaster to be built at the park since Blue Streak opened in 1964.

 Upon completion in 2000, Millennium Force broke six world records and was the world's first Giga Coaster, a roller coaster that exceeds 300 ft in height and
completes a full circuit.

 It was briefly the tallest and fastest in the world until Steel Dragon 2000 opened later the same year.

 The ride is also the third-longest roller coaster in North America after The Beast at Kings Island and Fury 325 at Carowinds.

 It was the first roller coaster to use a cable lift system rather than a traditional chain lift.

 The coaster has a 310 ft , 45-degree lift hill with a 300 ft drop and features two tunnels, three overbanked turns, and four hills.

 It has a top speed of 93 mph .
</item 2>

<item 3>
Space Mountain (Magic Kingdom)
Space Mountain is an indoor dark outer space-themed steel roller coaster at the Magic Kingdom theme park at the Walt Disney World Resort in Lake Buena Vista,
Florida.

 Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction
that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.

 RCA helped fund Space Mountain's construction and sponsored the ride from 1975 to 1993; FedEx sponsored Space Mountain from 1994 to 2004.
</item 3>

<item 4>
Space Mountain (Disneyland)
Space Mountain (known as Hyperspace Mountain from 2015 to 2017) is an indoor, space-themed roller coaster in Tomorrowland at Disneyland in Anaheim, California.

 Opened on May 27, 1977, it was the second roller coaster built at Disneyland, and was the second of the five versions of Space Mountain built by The Walt
Disney Company.

 Its exterior façade is one of Disneyland's four "mountain" structures that serve as park landmarks.
</item 4>

<item 5>
Walt Disney World Quest: Magical Racing Tour
Walt Disney World Quest: Magical Racing Tour is a 2000 go-kart racing video game based on attractions at the Walt Disney World Resort.

 Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain
Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.

 The game was developed by Crystal Dynamics and published by Eidos Interactive.

 The Game Boy Color version was developed by Prolific Publishing.
</item 5>

<item 6>
Rock 'n' Roller Coaster Starring Aerosmith
Rock 'n' Roller Coaster Starring Aerosmith is an enclosed launched steel roller coaster at Disney's Hollywood Studios at the Walt Disney World Resort and at the
Walt Disney Studios Park in Disneyland Paris (where it is called "Rock 'n' Roller Coaster avec Aerosmith").

 As the attraction's name suggests, the coaster features Aerosmith members, Steven Tyler, Joe Perry, Tom Hamilton, Joey Kramer, and Brad Whitford.
</item 6>

<item 7>
Space Mountain
Space Mountain is the name of a space-themed indoor roller coaster attraction located at five of the Magic Kingdom-style Disney Parks (all except for Shanghai
Disneyland Park).

 Although all five versions of the attraction are different in nature, all have a similar domed exterior façade that is a landmark for the respective park.

 The first Space Mountain ride opened in 1975 at Walt Disney World Resort in Florida, and similar attractions were then eventually built at the other Disney
parks.
</item 7>

<item 8>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 8>

<item 9>
Wild West Express Coaster
Wild West Express Coaster is a steel roller coaster operating at Glenwood Caverns Adventure Park in Glenwood Springs, Colorado.

 Wild West Express Coaster opened to the public at Glenwood Caverns Adventure Park on May 25, 2012.
</item 9>

<item 10>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 10>
=== END ITEM id=74a2178359ddbff3 ===
