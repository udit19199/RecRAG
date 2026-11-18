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

=== ITEM id=398e33d339768c49 ===
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
Park Avenue House
The Park Avenue House is a high rise residential building located at 2305 Park Avenue in the Park Avenue Historic District in Downtown Detroit, Michigan.

 It was listed on the National Register of Historic Places in 1996.

 It should not be confused with the nearby Park Avenue Hotel, which was demolished in 2015.
</item 1>

<item 2>
655 Park Avenue
655 Park Avenue is a Georgian-style co-op residential building on Manhattan's Upper East Side, located on Park Avenue between 67th Street and 68th Street,
adjacent to the Park Avenue Armory.

 It was developed in 1924 by Dwight P. Robinson & Company.

 The building at 655 Park Avenue was designed by architects James Edwin Ruthven Carpenter, Jr., often referred to by the initials "J.E.R. Carpenter", and Mott
B. Schmidt.

 Carpenter is considered the leading architect for luxury residential high-rise buildings in New York City in the early 1900s, while Schmidt is known for his
buildings in the American Georgian Classical style, including Sutton Place and houses for New York City's society figures and business elite.
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
MiMA (building)
MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen
neighborhood of Manhattan, New York City.

 Ground was broken in 2007 and topping out occurred in early August 2010.

 It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on
floors 51 to 63, and a Yotel hotel on the lower levels.

 The building's height is 204 meters (670 feet).
</item 4>

<item 5>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 5>

<item 6>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 6>

<item 7>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 7>

<item 8>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 8>

<item 9>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 9>

<item 10>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 10>
=== END ITEM id=398e33d339768c49 ===

=== ITEM id=22a486d3220b24d3 ===
Gold sentences and facts:
Sentence 0: Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
  Fact 0.0: The Spanish name of Teide National Park is 'Parque nacional del Teide'.
  Fact 0.1: Teide National Park is a national park.
  Fact 0.2: Teide National Park is located in Tenerife.
  Fact 0.3: Tenerife is in the Canary Islands, Spain.
Sentence 1: Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands (Spain).
  Fact 1.0: The Spanish name of Garajonay National Park is 'Parque nacional de Garajonay'.
  Fact 1.1: Garajonay National Park is located in the center of the island of La Gomera.
  Fact 1.2: Garajonay National Park is located in the north of the island of La Gomera.
  Fact 1.3: La Gomera is one of the Canary Islands in Spain.

Retrieved items:
<item 1>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 1>

<item 2>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 2>

<item 3>
Garajonay National Park, on La Gomera in Spain, covers 40 km², spans six municipalities, and became a national park in 1981 and UNESCO World Heritage Site in
1986. Bulgaria’s Pirin National Park covers 403.56 km², reaches 2,914 m at Vihren, and has been protected since 1962 and UNESCO-listed since 1983. Spain’s
Guadarrama National Park protects nearly 34,000
</item 3>

<item 4>
Landes de Gascogne Regional Natural Park
Landes de Gascogne Regional Natural Park (French: "Parc naturel régional des Landes de Gascogne") is a protected area of pine forest, wetland and oceanic
coastline located in the Aquitaine region of southwestern France.
</item 4>

<item 5>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 5>

<item 6>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 6>

<item 7>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 7>

<item 8>
Hermigua, on northeastern La Gomera, lies 12 km from the island capital; Garajonay National Park covers its south. Tenerife, the Canaries’ largest and most
populous island, contains Teide National Park and the emblematic Roque Cinchado. Guanche mummies preserve pre-Spanish indigenous remains; a Guanche pagan
organization was founded in La Laguna in 2001. Chile’s Cay is an apparently inactive stratovolcano; Penco
</item 8>

<item 9>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 9>

<item 10>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 10>
=== END ITEM id=22a486d3220b24d3 ===

=== ITEM id=328e7de331b49f1d ===
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
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 1>

<item 2>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
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
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 4>

<item 5>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 5>

<item 6>
F. W. Woolworth Building (Watertown, New York)
The Woolworth Building is an historic building in Watertown, New York.

 It is a contributing building in the Public Square Historic District.

 Plans for the Woolworth Building were begun in 1916 by Frank W. Woolworth, the founder of the Woolworth's chain of department stores.
</item 6>

<item 7>
Woolworth Building
The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US
skyscraper.

 The original site for the building was purchased by F. W. Woolworth and his real estate agent Edward J. Hogan by April 15, 1910, from the Trenor Luther Park
Estate and other owners for $1.65 million.

 By January 18, 1911, Woolworth and Hogan had acquired the final site for the project, totaling $4.5 million.

 More than a century after its construction, it remains, at 241.4 m , one of the 100 tallest buildings in the United States as well as one of the 30 tallest
buildings in New York City.

 It has been a National Historic Landmark since 1966, and a New York City landmark since 1983.
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
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 9>

<item 10>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 10>
=== END ITEM id=328e7de331b49f1d ===

=== ITEM id=eb693e99c9a9de48 ===
Gold sentences and facts:
Sentence 0: Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also in Halland.
  Fact 0.0: Spettekaka is also called spettkaka.
  Fact 0.1: In native Scanian, spettekaka is called "spiddekaga".
  Fact 0.2: Spettekaka is a local dessert.
  Fact 0.3: Spettekaka is a local dessert of the southern parts of Sweden.
  Fact 0.4: Spettekaka is chiefly found in the province of Scania (Skåne).
  Fact 0.5: Spettekaka is also found in Halland.
Sentence 1: Scanian (   ) is a closely related group of South Swedish dialects spoken in the province of Scania in southern Sweden.
  Fact 1.0: Scanian is a closely related group of South Swedish dialects.
  Fact 1.1: Scanian dialects are spoken in the province of Scania.
  Fact 1.2: Scanian dialects are spoken in southern Sweden.
Sentence 2:  Scanian formed part of the old Scandinavian dialect continuum and are by most historical linguists considered to be an East Danish dialect group, but due to the modern-era influence from Standard Swedish in the region and because traditional dialectology in the Scandinavian countries normally has not considered isoglosses that cut across state borders, the Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.
  Fact 2.0: Scanian formed part of the old Scandinavian dialect continuum.
  Fact 2.1: Most historical linguists consider Scanian to be an East Danish dialect group.
  Fact 2.2: Standard Swedish has influenced the Scanian dialects in the region in the modern era.
  Fact 2.3: The Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.
  Fact 2.4: The Scanian dialects have been treated as South Swedish due to the modern-era influence from Standard Swedish in the region.
  Fact 2.5: The Scanian dialects have been treated as South Swedish because traditional dialectology in the Scandinavian countries normally has not considered isoglosses that cut across state borders.

Retrieved items:
<item 1>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 1>

<item 2>
Hakata dialect
Hakata dialect (博多弁 , Hakata-ben ) is a Japanese dialect spoken in Fukuoka city.

 Hakata dialect originated in Hakata commercial district, while a related Fukuoka dialect (福岡弁 , Fukuoka-ben ) was spoken in the central district.

 Hakata dialect has spread throughout the city and its suburbs.

 Most of Japanese regard Hakata dialect as the dialect typical of Fukuoka, so it is sometimes called "Fukuoka-ben".
</item 2>

<item 3>
Scanian dialect
Scanian (   ) is a closely related group of South Swedish dialects spoken in the province of Scania in southern Sweden.

 Scanian formed part of the old Scandinavian dialect continuum and are by most historical linguists considered to be an East Danish dialect group, but due to
the modern-era influence from Standard Swedish in the region and because traditional dialectology in the Scandinavian countries normally has not considered
isoglosses that cut across state borders, the Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.

 However, many of the early Scandinavian linguists, including Adolf Noreen and G. Sjöstedt, classified it as "South Scandinavian", and some linguists, such as
Elias Wessén, also considered Old Scanian a separate language, classified apart from both Old Danish and Old Swedish.
</item 3>

<item 4>
Graph query row:
dialect: Scania
spettekaka_relationship: LOCATED_IN
scania_relationship: LOCATED_IN
</item 4>

<item 5>
The Saga of Gosta Berling
The Saga of Gosta Berling (Swedish: Gösta Berlings saga ) is a 1924 Swedish romantic drama film directed by Mauritz Stiller and released by AB Svensk
Filmindustri, starring Lars Hanson, Gerda Lundequist and Greta Garbo in her native break-out role on film.

 The film is based on the 1891 debut novel of the same name by the Swedish author and Nobel Prize winner Selma Lagerlöf.

 It is also known as Gosta Berling's Saga, The Story of Gosta Berling and The Atonement of Gosta Berling.
</item 5>

<item 6>
South Bergish
South Bergish (German: "Südbergische Dialekte ") or Upper Bergish (German: "Oberbergische Dialekte ") is a group of German dialects of the Bergisches Land
Region East of the Rhine and approximately south of the Wupper and north of the Sieg.

 These dialects are part of the Ripuarian group and thus are also called East Ripuarian.

 Ripuarian dialects are also spoken west of the Rhine up to the German border, and in some small areas next to the respective borders in Belgium and in the
Netherlands.

 Ripuarian Bergish dialects belong to the Middle German group, and thus are varieties of High German, where they belong to the northmost ones.

 In the North, they border to the East Bergish and the West Bergish aka Low Bergish language groups, which are part of the Low Franconian group like Dutch.
</item 6>

<item 7>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 7>

<item 8>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 8>

<item 9>
Spettekaka
Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also
in Halland.

 It is an important part of the Scanian culinary heritage.

 The name means "cake on a spit", and this describes the method of preparation: it is the Swedish variation on the spit cake.
</item 9>

<item 10>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 10>
=== END ITEM id=eb693e99c9a9de48 ===

=== ITEM id=e5ac19fe39228522 ===
Gold sentences and facts:
Sentence 0: Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.
  Fact 0.0: Edward Benjamin Britten, Baron Britten, lived from 22 November 1913 to 4 December 1976.
  Fact 0.1: Edward Benjamin Britten, Baron Britten, was an English composer.
  Fact 0.2: Edward Benjamin Britten, Baron Britten, was an English conductor.
  Fact 0.3: Edward Benjamin Britten, Baron Britten, was an English pianist.
Sentence 1: Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.
  Fact 1.0: Béla Viktor János Bartók lived from 25 March 1881 to 26 September 1945.
  Fact 1.1: Béla Viktor János Bartók was a Hungarian composer.
  Fact 1.2: Béla Viktor János Bartók was a Hungarian pianist.
  Fact 1.3: Béla Viktor János Bartók was a Hungarian ethnomusicologist.

Retrieved items:
<item 1>
Béla Bartók Music High School
Béla Bartók Music High School (Bartók Béla Zeneművészeti Szakközépiskola) is situated in the Palace of Music (Zenepalota) in Bartók square Miskolc, Hungary.

It is a famous Music school named after the famous Hungarian composer Béla Bartók.

It was founded in 1966.
</item 1>

<item 2>
Benjamin Britten
Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.

 He was a central figure of 20th-century British classical music, with a range of works including opera, other vocal music, orchestral and chamber pieces.

 His best-known works include the opera "Peter Grimes" (1945), the "War Requiem" (1962) and the orchestral showpiece "The Young Person's Guide to the Orchestra"
(1945).
</item 2>

<item 3>
Suite, Op. 14 (Bartók)
The Suite, Op. 14, Sz.

 62, BB 70 is a piece for solo piano written by Béla Bartók.

 It was written in February 1916, published in 1918, and debuted by the composer on April 21, 1919, in Budapest.

 The Suite is one of Bartók's most significant works for piano, only comparable with his 1926 Piano Sonata.

 Though much of Bartók's work makes frequent use of Eastern European folk music, this suite is one of the few pieces without melodies of folk origin.

 However, Romanian, Arabic, and North African rhythmic influences can still be found in some movements.

 Originally intending the suite to be a five-movement work, Bartók later decided against the idea and discarded the second movement, the "Andante", which was
published only posthumously in the October 1955 issue of "Új Zenei Szemle" (New Musical Review).
</item 3>

<item 4>
String Quartet No. 6 (Bartók)
Due to the difficulties of the war, communication between Bartók and Székely was difficult, and the quartet was not premiered until 20 January 1941, when the
Kolisch Quartet, to whom the work is dedicated, gave its premiere at the Town Hall in New York City.
</item 4>

<item 5>
List of compositions by Béla Bartók
This aspires to be a complete list of compositions by Béla Bartók.

 The catalogue numbering by András Szőllősy (Sz.)

, László Somfai (BB.)

 and Denijs Dille (DD.)

 are provided, as well as Bartók's own opus numbers.

 Note that Bartók started three times anew with opus numbers, here indicated with "(list 1)", "(list 2)" and "(list 3)" respectively.

 The pieces from the third listing are by far best known; opus lists 1 and 2 are early works.

 The year of composition and instrumentation (including voice) are included.

 See the main article on Béla Bartók for more details.
</item 5>

<item 6>
Bluebeard's Castle
Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.

 The libretto was written by Béla Balázs, a poet and friend of the composer, and is written in Hungarian, based on the French literary tale "La Barbe bleue" by
Charles Perrault.

 The opera lasts only a little over an hour and there are only two singing characters onstage: Bluebeard ("Kékszakállú "), and his new wife Judith ("Judit " );
the two have just eloped and Judith is coming home to Bluebeard's castle for the first time.
</item 6>

<item 7>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 7>

<item 8>
List of string quartets by Béla Bartók
1 and 28 both owe a great deal to Bartók's quartets (; ), György Ligeti, whose two string quartets both owe a great deal to Bartók’s quartets (; ), Bruno
Maderna , George Perle, who credits the Bartók Fourth and Fifth Quartets as precedents for his use of arrays of chords related to one another by different types
of symmetry , Walter Piston (; ), Kim Dzmitrïyevich Tsesakow , Wilfried Westerlinck , Stefan Wolpe, who explained in a public lecture how he had derived ideas
from Bartók’s Fourth Quartet , and Xu Yongsan .
</item 8>

<item 9>
String Quartet No. 6 (Bartók)
The String Quartet No. 6, Sz.

 114, BB 119, was the final string quartet that Béla Bartók wrote before his death.

 It was begun in August 1939 in Saanen, Switzerland, where Bartók was a guest of his patron, the conductor Paul Sacher.

 Shortly after he completed the Divertimento for String Orchestra on the 17th, he started on a commission for his friend, the violinist Zoltán Székely.

 Székely was acting as intermediary for the "New Hungarian Quartet", who had given the Budapest premiere of the String Quartet No. 5.

 With the outbreak of World War II and his mother's illness, Bartók returned to Budapest, where the quartet was finished in November.

 After his mother's death, Bartók decided to leave with his family for the United States.
</item 9>

<item 10>
List of string quartets by Béla Bartók
The Hungarian composer Béla Bartók wrote six string quartets, each for the usual forces of two violins, viola and cello.

 Notable composers who have been influenced by them include Benjamin Britten, particularly in the Sonata in C for Cello and Piano (; ), Elliott Carter, who
refers in the opening of his own First String Quartet to Bartók’s Sixth Quartet , Chen Yi , Edison Denisov, whose Second Quartet is closely related to Bartók’s
Fifth Quartet , Franco Donatoni, who was deeply impressed when he heard a broadcast of Bartók's Fourth Quartet , Robert Fripp, who mentions them as an influence
upon King Crimson , Miloslav Ištvan , György Kurtág, whose Opp.
</item 10>
=== END ITEM id=e5ac19fe39228522 ===

=== ITEM id=110486f3f73c80bb ===
Gold sentences and facts:
Sentence 0: Ecopack Ltd is a PET bottles blowing company in Pakistan.
  Fact 0.0: Ecopack Ltd is a PET bottles blowing company.
  Fact 0.1: Ecopack Ltd is located in Pakistan.
Sentence 1: Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and in combination with glass fiber for engineering resins.
  Fact 1.0: Polyethylene terephthalate is also written as poly(ethylene terephthalate).
  Fact 1.1: Polyethylene terephthalate is commonly abbreviated as PET.
  Fact 1.2: Polyethylene terephthalate is commonly abbreviated as PETE.
  Fact 1.3: Polyethylene terephthalate is sometimes abbreviated as PETP, which is obsolete.
  Fact 1.4: Polyethylene terephthalate is sometimes abbreviated as PET-P, which is obsolete.
  Fact 1.5: Polyethylene terephthalate is the most common thermoplastic polymer resin of the polyester family.
  Fact 1.6: Polyethylene terephthalate is used in fibers for clothing.
  Fact 1.7: Polyethylene terephthalate is used in containers for liquids and foods.
  Fact 1.8: Polyethylene terephthalate is used in thermoforming for manufacturing.
  Fact 1.9: Polyethylene terephthalate is used in combination with glass fiber for engineering resins.

Retrieved items:
<item 1>
Polyethylene 2,5-furandicarboxylate
Polyethylene 2,5-furandicarboxylate, also named poly(ethylene 2,5-furandicarboxylate), polyethylene furanoate and poly(ethylene furanoate) and generally
abbreviated as PEF, is a polymer that can be produced by polycondensation of 2,5-furandicarboxylic acid (FDCA) and monoethylene glycol (MEG).

 As an aromatic polyester from ethylene glycol it is a chemical analogue of polyethylene terephthalate (PET) and polyethylene naphthalate (PEN).

 PEF has been described in (patent) literature since 1951, but has gained renewed attention since the US department of energy proclaimed its building block,
FDCA, as a potential bio-based replacement for purified terephthalic acid (PTA) in 2004.
</item 1>

<item 2>
BoPET
BoPET (Biaxially-oriented polyethylene terephthalate) is a polyester film made from stretched polyethylene terephthalate (PET) and is used for its high tensile
strength, chemical and dimensional stability, transparency, reflectivity, gas and aroma barrier properties, and electrical insulation.
</item 2>

<item 3>
Ecopack
Ecopack Ltd is a PET bottles blowing company in Pakistan.
</item 3>

<item 4>
Xenoy
Xenoy is a blend of plastics with many industrially-useful properties.

 It is typically polyester (polybutylene terephthalate, PBT, or polyethylene terephthalate, PET) and polycarbonate (PC): it is often labeled PBT+PC or PET+PC.
</item 4>

<item 5>
Two-liter bottle
The two-liter bottle is a common container for soft drinks.

 These bottles are produced from polyethylene terephthalate, also known as PET plastic, using the blow molding process.

 Bottle labels consist of a printed, tight-fitted plastic sleeve.

 A resealable screw-top allows the contents to be used at various times while retaining carbonation.
</item 5>

<item 6>
Polytrimethylene terephthalate
Polytrimethylene terephthalate (PTT), is a polyester synthesized and patented in 1941.

 It is produced by a method called condensation polymerization or transesterification.

 The two monomer units used in producing this polymer are: 1,3-propanediol and terephthalic acid or dimethyl terephthalate.

 Similar to polyethylene terephthalate, the PTT is used to make carpet fibers.
</item 6>

<item 7>
Polyethylene naphthalate
Polyethylene naphthalate (poly(ethylene 2,6-naphthalate or PEN) is a polyester with good barrier properties (even better than Polyethylene terephthalate).

 Because it provides a very good oxygen barrier, it is particularly well-suited for bottling beverages that are susceptible to oxidation, such as beer.

 It is also used in making high performance sailcloth.

 It also has been found to show supreme scintillation properties and is expected to replace classic plastic scintillators.
</item 7>

<item 8>
Krones
Krones AG is a German packaging and bottling machine manufacturer.

 It is the world's leading manufacturer of lines for filling beverages in plastic and glass bottles or beverage cans.

 The company manufactures stretch blow-moulding machines for producing polyethylene terephthalate (PET) bottles, plus fillers, labellers, bottle washers,
pasteurisers, inspectors, packers and palletisers.

 This product portfolio is complemented by material flow systems and process technology for producing beverages, plus syrup kitchens, for clients like
breweries, dairies and soft-drink companies.
</item 8>

<item 9>
Polyethylene terephthalate
Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common
thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and
in combination with glass fiber for engineering resins.
</item 9>

<item 10>
Polycyclohexylenedimethylene terephthalate
Polycyclohexylenedimethylene terephthalate (PCT) is a thermoplastic polyester formed from the polycondensation of terephthalic acid and cyclohexanedimethanol.

Its chemical structure is similar to that of polyethylene terephthalate (PET), with which it shares properties like dimensional stability and chemical
resistance.

 PCT is also particularly resistant to high temperatures and hydrolysis.

 The melting point is 545 °F (285 °C).

 Common brand names are "Thermx" (Ticona), "Eastar" (Eastman) and "SkyPURA" (SK Chemicals).
</item 10>
=== END ITEM id=110486f3f73c80bb ===

=== ITEM id=18684059caf8103c ===
Gold sentences and facts:
Sentence 0: Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities (with notable exceptions).
  Fact 0.0: Kresy Wschodnie, also called Kresy, Eastern Borderlands, or Borderlands, was a region of the Second Polish Republic during the interwar period.
  Fact 0.1: Kresy constituted nearly half of the territory of the Second Polish Republic.
  Fact 0.2: In Kresy, ethnic Poles were the largest group.
  Fact 0.3: In Kresy, ethnic Poles were roughly equal in number to the size of the national minorities.
  Fact 0.4: The rough equality of ethnic Poles with the national minorities in Kresy had notable exceptions.
Sentence 1:  It was drawn for the first time by the Supreme War Council as the demarcation line between the newly emerging states, the Second Polish Republic, and the Soviet Union.
  Fact 1.0: Kresy was drawn for the first time by the Supreme War Council.
  Fact 1.1: Kresy was drawn as the demarcation line between the newly emerging states, the Second Polish Republic and the Soviet Union.

Retrieved items:
<item 1>
Tripartite Indenture
The Tripartite Indenture was an agreement made in February 1405 between Owain Glyndŵr, Edmund Mortimer, and Henry Percy, 1st Earl of Northumberland, agreeing to
divide England and Wales up between them at the expense of Henry IV.

 Glyndŵr was to be given Wales, and a substantial part of the west of England, including the English portions of the Welsh Marches.

 Northumberland was to have received the north, as well as Northamptonshire, Norfolk, Warwickshire, and Leicestershire.

 The Mortimers were to have received the rest of southern England.
</item 1>

<item 2>
Schlieffen Plan
The Schlieffen Plan (German: "Schlieffen-Plan" , ] ) was the name given after World War I to the thinking behind the German invasion of France and Belgium on 4
August 1914.

 Field Marshal Alfred von Schlieffen, the Chief of the Imperial Army German General Staff from 1891 to 1906, devised a deployment plan for a war-winning
offensive, in a one-front war against the French Third Republic from 1905–06.

 After the war, the German official historians of the "Reichsarchiv" and other writers, described the plan as a blueprint for victory.

 German historians claimed that the plan had been ruined by "Generaloberst" (Colonel-General) Helmuth von Moltke the Younger, the Commander-in-Chief of the
German army after Schlieffen retired in 1906, who was dismissed after the First Battle of the Marne (5–12 September 1914).
</item 2>

<item 3>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 3>

<item 4>
Curzon Line
The history of the Curzon Line, with minor variations, goes back to the period following World War I.

 It was drawn for the first time by the Supreme War Council as the demarcation line between the newly emerging states, the Second Polish Republic, and the
Soviet Union.

 The proposal was put forward by British Foreign Secretary George Curzon, to serve as a diplomatic basis for the future border agreement, and in that form, it
never materialized because the war went on.
</item 4>

<item 5>
Kresy
In September 1939, after the Soviet Union joined Nazi Germany in their attack on Poland in accordance with the Molotov–Ribbentrop Pact, the territories were
incorporated into Soviet Ukraine, Belarus and Lithuania in the atmosphere of terror.
</item 5>

<item 6>
Kresy
Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly
half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities
(with notable exceptions).

 Administratively, the territory of Kresy was composed of voivodeships of Lwów, Nowogródek, Polesie, Stanisławów, Tarnopol, Wilno, Wołyń, and the Białystok.

 Today, these territories are divided between Western Ukraine, Western Belarus, and south-eastern Lithuania, with such major cities as Lviv, Vilnius, and Grodno
no longer in Poland.

 In the Second Polish Republic the term "Kresy" roughly equated with the lands beyond the so-called Curzon Line, which was suggested after World War I in
December 1919 by the British Foreign Office as the eastern border of the re-emerging sovereign Republic following the century of partitions.
</item 6>

<item 7>
New York – New Jersey Line War
The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between
1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
</item 7>

<item 8>
Lac La Belle and Calumet Railroad
The Lac La Belle and Calumet Railroad was an American, narrow gauge railroad that operated in the Keweenaw Peninsula, or the extreme northern Upper Peninsula of
Michigan.

 The line ran between a stamp mill at Lac La Belle and two copper mines, the Mendota and the Delaware, from 1883 to 1888, when poor economic conditions forced
the line's closure.

 The defunct company and its tracks were later purchased by the Copper Range Company, which converted the line to standard gauge and changed the railroad's name
to the Keweenaw Central.

 This company failed in 1917, and the former Lac La Belle and Calumet tracks were removed in 1918.
</item 8>

<item 9>
Battle of Berestechko
The Battle of Berestechko (Polish: "Bitwa pod Beresteczkiem" ; Ukrainian: Берестецька битва, Битва під Берестечком ) was fought between the Ukrainian Cossacks,
led by Hetman Bohdan Khmelnytsky, aided by their Crimean Tatar allies, and a Polish army under King John II Casimir.

 It was a battle of a Cossack rebellion in Ukraine that took place in the years 1648–1657 after the expiration of a two-year truce.

 Fought from 28 to 30 June 1651, the battle took place in the province of Volhynia, on the hilly plain south of the Styr River.

 The Polish camp was on the river opposite Berestechko and faced south, towards the Cossack army about two kilometers away, whose right flank was against the
River Pliashivka (Pliashova) and the Tatar army on their left flank.

 It was probably the largest European land battle of the 17th century.
</item 9>

<item 10>
Transoxiana
Transoxiana (also spelled Transoxania), known in Arabic sources as Mā warāʼ al-Nahr (Arabic: ما وراء النهر‎ ‎ ] – 'what [is] beyond the [Oxus] river') and in
Persian as Farārūd (Persian: فرارود‎ ‎ , ] —'beyond the [Amudarya] river'), is the ancient name used for the portion of Central Asia corresponding approximately
with modern-day Uzbekistan, Tajikistan, southern Kyrgyzstan, and southwest Kazakhstan.

 Geographically, it is the region between the Amu Darya (Ancient Greek: Ώξος "Ốxos") and Syr Darya rivers.

 The area had been known to the Romans as Transoxania (Land beyond the Oxus), to the Arabs as Mawarannahr (Land Beyond the River), and to the Iranians as Turan,
a term used in the Persian national epic Shahnameh.
</item 10>
=== END ITEM id=18684059caf8103c ===

=== ITEM id=2492b280e7496703 ===
Gold sentences and facts:
Sentence 0: Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.
  Fact 0.0: Marcel Émile Verdet was born on 13 March 1824.
  Fact 0.1: Marcel Émile Verdet died on 3 June 1866.
  Fact 0.2: Marcel Émile Verdet was a French physicist.
Sentence 1:  Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de physique".
  Fact 1.0: Marcel Émile Verdet did much to champion the early theory of the conservation of energy in France.
  Fact 1.1: Marcel Émile Verdet championed this theory in France through his editorial supervision of the 'Annales de chimie et de physique'.
Sentence 2: Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the title "Annales de chimie".
  Fact 2.0: Annales de chimie et de physique is French for 'Annals of Chemistry and of Physics'.
  Fact 2.1: Annales de chimie et de physique is a scientific journal.
  Fact 2.2: Annales de chimie et de physique was founded in Paris, France, in 1789.
  Fact 2.3: Annales de chimie et de physique was founded under the title 'Annales de chimie'.

Retrieved items:
<item 1>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 1>

<item 2>
The Picture of Dorian Gray
The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".

 The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.

 Despite that censorship, "The Picture of Dorian Gray" offended the moral sensibilities of British book reviewers, some of whom said that Oscar Wilde merited
prosecution for violating the laws guarding the public morality.

 In response, Wilde aggressively defended his novel and art in correspondence with the British press, although he personally made excisions of some of the most
controversial material when revising and lengthening the story for book publication the following year.
</item 2>

<item 3>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 3>

<item 4>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 4>

<item 5>
Index of Middle English Verse
The Index of Middle English Verse (IMEV) is a bibliographic index of poetry in Middle English.

 Its first print publication, in 1943, was an extension of Carleton Brown's "Register of Middle English Religious & Didactic Verse", augmented by the inclusion
of secular verse.

 This edition, edited by Brown and Rossell Hope Robbins, contained entries for over 4000 Middle English poems in more than 2000 manuscripts.

 In 1965 the index was supplemented by Robbins and John L. Cutler.

 The IMEV is available online through Virginia Polytechnic Institute and State University.
</item 5>

<item 6>
The Henry James Review
The Henry James Review is a triannual peer-reviewed academic journal established in 1979 and is the official publication of the Henry James Society.

 It is dedicated to the scholarly, critical, and theoretical study of the American writer Henry James.

 Each issue focuses on a specific theme of interest and seeks to promote understanding and study of James' contributions.

 The journal is published by the Johns Hopkins University Press and the current editor-in-chief is Susan M. Griffin (University of Louisville).
</item 6>

<item 7>
Around the World in Eighty Days
Around the World in Eighty Days (French: "Le tour du monde en quatre-vingts jours" ) is a classic adventure novel by the French writer Jules Verne, published in
1873.

 In the story, Phileas Fogg of London and his newly employed French valet Passepartout attempt to circumnavigate the world in 80 days on a £20,000 wager (the
approximate equivalent of £2 million in 2016) set by his friends at the Reform Club.

 It is one of Verne's most acclaimed works .
</item 7>

<item 8>
La Belle Assemblée
La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine
published from 1806 to 1837, founded by John Bell (1745–1831).
</item 8>

<item 9>
Entente Cordiale (opera)
Entente Cordiale is a comic opera in one act by Ethel Smyth with an English-language libretto by Smyth, who describes the work as "a post-war comedy in one act
(founded on fact)".

 It was first performed by students at the Royal College of Music in London on 22 July 1925.
</item 9>

<item 10>
Guglielmo Verdirame
Guglielmo Verdirame (born in Reggio di Calabria, Italy) is a Professor of International Law at King's College London in the Department of War Studies and the
School of Law.

 He was previously a university lecturer in law at the University of Cambridge, Faculty of Law, and a Fellow of the Lauterpacht Centre for International Law.

 He practises as a barrister at 20 Essex Street Chambers.
</item 10>
=== END ITEM id=2492b280e7496703 ===

=== ITEM id=ad0624e02a274b46 ===
Gold sentences and facts:
Sentence 0: That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.
  Fact 0.0: That Darn Cat is a film.
  Fact 0.1: That Darn Cat is from 1997.
  Fact 0.2: That Darn Cat is American.
  Fact 0.3: That Darn Cat is a mystery comedy.
  Fact 0.4: That Darn Cat stars Christina Ricci.
  Fact 0.5: That Darn Cat stars Doug E. Doug.
Sentence 1: Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".
  Fact 1.0: Honey, I Blew Up the Kid is a film.
  Fact 1.1: Honey, I Blew Up the Kid is from 1992.
  Fact 1.2: Honey, I Blew Up the Kid is American.
  Fact 1.3: Honey, I Blew Up the Kid is a comedy science fiction film.
  Fact 1.4: Honey, I Blew Up the Kid is a sequel to Honey, I Shrunk the Kids.
  Fact 1.5: Honey, I Shrunk the Kids is a 1989 film.

Retrieved items:
<item 1>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 1>

<item 2>
Honey, We Shrunk Ourselves
Honey, We Shrunk Ourselves is a 1997 live-action direct-to-video sequel to "Honey, I Shrunk the Kids" and "Honey, I Blew Up the Kid".

 It is the third and final installment in the "Honey, I Shrunk the Kids" trilogy.

 The directorial debut of cinematographer Dean Cundey and released through Walt Disney Home Video, it tells the story of inventor Wayne Szalinski as he
accidentally shrinks his wife, brother, sister-in-law, and himself with his electromagnetic shrink ray.
</item 2>

<item 3>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 3>

<item 4>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 4>

<item 5>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 5>

<item 6>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 6>

<item 7>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 7>

<item 8>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 8>

<item 9>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 9>

<item 10>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 10>
=== END ITEM id=ad0624e02a274b46 ===

=== ITEM id=9df7657ea54eba2c ===
Gold sentences and facts:
Sentence 0: "Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series overall.
  Fact 0.0: Moho House is the twenty-first episode of the twenty-eighth season of the animated television series The Simpsons.
  Fact 0.1: Moho House is the 617th episode of The Simpsons overall.
  Fact 0.2: The Simpsons is an animated television series.
Sentence 1:  On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
  Fact 1.0: Fox announced on May 4, 2015.
  Fact 1.1: Fox announced that The Simpsons had been renewed for season 28.

Retrieved items:
<item 1>
List of The Simpsons comics
The following is a list of comic book series based on the animated TV show The Simpsons and published by Bongo Comics in the United States.

 The first comic strips based on "The Simpsons" appeared in 1991 in the magazine "Simpsons Illustrated" (not to be confused with the comic publications from
2012 bearing the same name), which was a companion magazine to the show.

 The comic strips were popular and a one-shot comic book entitled "Simpsons Comics and Stories", containing three different stories, was released in 1993 for
the fans.

 The book was a success and due to this, the creator of "The Simpsons", Matt Groening, and his companions Bill Morrison, Mike Rote, Steve Vance and Cindy Vance
created the publishing company Bongo Comics.

 By the end of 1993, Bongo was publishing four titles: "Simpsons Comics", "Bartman", "Radioactive Man" and "Itchy & Scratchy Comics".
</item 1>

<item 2>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 2>

<item 3>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 3>

<item 4>
The Simpsons (season 28)
The twenty-eighth season of the animated television series "The Simpsons" began airing on Fox in the United States on September 25, 2016, and ended on May 21,
2017.

 On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
</item 4>

<item 5>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 5>

<item 6>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 6>

<item 7>
List of The Simpsons comics
Since then, many more titles have been published, out of which "Simpsons Comics", "Bart Simpson", "Bart Simpson's Treehouse of Horror", "Simpsons Super
Spectacular", Simpsons Summer Shindig, and "Simpsons Winter Wingding".
</item 7>

<item 8>
The Simpsons Spin-Off Showcase
"The Simpsons Spin-Off Showcase" is the twenty-fourth episode of the eighth season of "The Simpsons".

 It originally aired on the Fox network in the United States on May 11, 1997.

 The episode centers on fictional pilot episodes of non-existent television series derived from "The Simpsons", and is a parody of the tendency of networks to
spin off characters from a hit series.

 As such it includes references to many different TV series.

 The first fictional spin-off is "Chief Wiggum P.I.", a cop-drama featuring Chief Wiggum and Seymour Skinner.

 The second is "The Love-matic Grampa", a sitcom featuring Moe Szyslak who receives dating advice from Abraham Simpson, whose ghost is possessing a love testing
machine.

 The final segment is "The Simpson Family Smile-Time Variety Hour", a variety show featuring the Simpson family except for Lisa, who has been replaced.
</item 8>

<item 9>
The Simpsons Theme
"The Simpsons" Theme", also referred to as "The Simpsons" Main Title Theme" in album releases, is the theme music of the animated television series "The
Simpsons".

 It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style
theme.

 The piece, which took 3 days, 2 hours, 48 minutes, and 19 seconds to create, has been noted by Elfman as the most popular of his career.

 The theme, as used for the opening sequence, was re-arranged during season 2, and the current arrangement by Alf Clausen was introduced at the beginning of the
third season.
</item 9>

<item 10>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 10>
=== END ITEM id=9df7657ea54eba2c ===

=== ITEM id=ab1e1e0e2b4a9dca ===
Gold sentences and facts:
Sentence 0: Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York, United States.
  Fact 0.0: Albany International Airport has the IATA code ALB.
  Fact 0.1: Albany International Airport has the ICAO code KALB.
  Fact 0.2: Albany International Airport has the FAA LID ALB.
  Fact 0.3: Albany International Airport is a public airport.
  Fact 0.4: Albany International Airport is seven miles (11 km) northwest of Albany.
  Fact 0.5: Albany International Airport is in Albany County, New York, United States.
Sentence 1: Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown Madison, the capital of Wisconsin.
  Fact 1.0: Dane County Regional Airport (DCRA) has the IATA code MSN.
  Fact 1.1: Dane County Regional Airport (DCRA) has the ICAO code KMSN.
  Fact 1.2: Dane County Regional Airport (DCRA) has the FAA LID MSN.
  Fact 1.3: Dane County Regional Airport is also known as Truax Field.
  Fact 1.4: Dane County Regional Airport is a civil-military airport.
  Fact 1.5: Dane County Regional Airport is located six miles northeast of downtown Madison.
  Fact 1.6: Madison is the capital of Wisconsin.

Retrieved items:
<item 1>
Air Wisconsin
Air Wisconsin Airlines is a regional airline based at Appleton International Airport in the town of Greenville, Wisconsin, United States, near Appleton.

 Air Wisconsin is the largest privately held regional airline in the United States.

 It currently operates regional jet flights as American Eagle under contract to American Airlines via a code sharing agreement, serving cities in the U.S. and
Canada with hubs at Boston Logan International Airport (BOS), Charlotte Douglas International Airport (CLT), Philadelphia International Airport (PHL) and Ronald
Reagan Washington National Airport (DCA) .

 Air Wisconsin previously operated United Express service on behalf of United Airlines followed by US Airways Express service on behalf of US Airways prior to
becoming an American Eagle regional air carrier.

 The company has announced it will once again partner with United Airlines as a United Express code sharing air carrier with primary hubs to be located at
Chicago O'Hare International Airport (ORD) and Washington Dulles International Airport (IAD) beginning in September 2017 and will be "exclusively operating" as
United Express by March 2018.
</item 1>

<item 2>
Dane County Regional Airport
Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown
Madison, the capital of Wisconsin.

 In the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, it is categorized as a small-hub primary commercial
service facility.
</item 2>

<item 3>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 3>

<item 4>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 4>

<item 5>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 5>

<item 6>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 6>

<item 7>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 7>

<item 8>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 8>

<item 9>
Truax Field Air National Guard Base
Truax Field Air National Guard Base (IATA: MSN, ICAO: KMSN, FAA LID: MSN) , also known as Truax Field, is a military facility located at Dane County Regional
Airport.

 It is located five miles (8 km) northeast of the center of Madison, a city in Dane County, Wisconsin, United States.
</item 9>

<item 10>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 10>
=== END ITEM id=ab1e1e0e2b4a9dca ===
