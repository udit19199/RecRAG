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

=== ITEM id=9ab7c6c4ac072ef2 ===
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
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 1>

<item 2>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
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
West Street District
The West Street District is a historic district on West Street in Boston, Massachusetts, one of the city's "ladder districts" that runs between Tremont Street
and Washington Street in the Downtown Crossing commercial/retail area.

 The district includes four buildings located near the corner of Tremont and West Streets, all built in the early 20th century.

 The two buildings at 148-150 Tremont Street were once occupied by Chandler and Company, an exclusive department store.

 Number 148 is a Renaissance Revival structure built as an office building in 1912, and number 150 was built in 1903 to house the Oliver Ditson Company, a music
publisher.

 The Fabyan building at 26-30 West Street was designed by Coolidge, Shepley, Bulfinch & Abbott, and built in 1926.

 The Schraffts Building at 16-24 West Street was built in 1922, and housed a flagship candy store and restaurant for more than fifty years.
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
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 7>

<item 8>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 8>

<item 9>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 9>

<item 10>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 10>
=== END ITEM id=9ab7c6c4ac072ef2 ===

=== ITEM id=4c4c52c1cec20c60 ===
Gold sentences and facts:
Sentence 0: Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.
  Fact 0.0: Underdog is a 2007 American film.
  Fact 0.1: Underdog is a family superhero comedy film.
  Fact 0.2: Underdog is based on the 1960s cartoon series of the same name.
Sentence 1: Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi, a Life in the Woods" by Austrian author Felix Salten.
  Fact 1.0: Bambi is a 1942 American animated film.
  Fact 1.1: Bambi was directed by David Hand.
  Fact 1.2: David Hand supervised a team of sequence directors on Bambi.
  Fact 1.3: Bambi was produced by Walt Disney.
  Fact 1.4: Bambi is based on the book 'Bambi, a Life in the Woods'.
  Fact 1.5: The book 'Bambi, a Life in the Woods' was written by Austrian author Felix Salten.

Retrieved items:
<item 1>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 1>

<item 2>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 2>

<item 3>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 3>

<item 4>
Bambi
Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi,
a Life in the Woods" by Austrian author Felix Salten.

 The film was released by RKO Radio Pictures on August 13, 1942, and is the fifth Disney animated feature film.
</item 4>

<item 5>
Get a Horse!
Get a Horse!

 is a 2013 American 3D animated slapstick comedy short film, produced by Walt Disney Animation Studios.

 Combining black-and-white hand-drawn animation and color CGI animation, the short features the characters of the late 1920s "Mickey Mouse" cartoons, and
features archival recordings of Walt Disney in his posthumous role as Mickey Mouse.

 It is the first original Mickey Mouse theatrical animated short since "Runaway Brain" (1995), and the first appearance of Oswald the Lucky Rabbit in a Disney
animated production in 85 years.
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
Bosko the Musketeer
Bosko the Musketeer is an American animated short film.

 It is a "Looney Tunes" cartoon, featuring Bosko, the first star of the series.

 It was released on August 12, 1933, although some sources note September 16 as a date; this is problematic, as that would imply that the last films featuring
Bosko as the star of Warner Bros. cartoons were released after the first film featuring Buddy, the second star of the series.

 It was, like most "Looney Tunes" of the time, directed by Hugh Harman; Frank Marsales was the musical director.
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
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 9>

<item 10>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 10>
=== END ITEM id=4c4c52c1cec20c60 ===

=== ITEM id=ce8c01fed92fc4b5 ===
Gold sentences and facts:
Sentence 0: 'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.
  Fact 0.0: A Guarani word, written in the source as 'aba], means 'what gives strength to the Indian'.
  Fact 0.1: The Guarani word is used for the infusions of the bark of a number of trees native to Brazil.
Sentence 1: Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South America that belongs to the Tupi–Guarani family of the Tupian languages.
  Fact 1.0: Guarani specifically refers to the primary variety known as Paraguayan Guarani.
  Fact 1.1: Paraguayan Guarani has the endonym avañe'ẽ, meaning 'the people's language'.
  Fact 1.2: Guarani is an indigenous language of South America.
  Fact 1.3: Guarani belongs to the Tupi–Guarani family of the Tupian languages.

Retrieved items:
<item 1>
San Antonio River (South America)
The San Antonio River (Spanish, Río San Antonio, Portuguese, Rio Santo Antônio; also called San Antonio Guazú) is a tributary of the Iguazu River.

 The San Antonio River forms the border between Misiones Province in Argentina and Paraná State in Brazil.

 South of the San Antonio's source near Barracão, the international border continues south along the Pepiri-Guazu River, which forms the border between Misiones
Province and Santa Catarina State.
</item 1>

<item 2>
Cora language
Cora is an indigenous language of Mexico of the Uto-Aztecan language family.

 It is spoken by the ethnic group that is widely known as the Cora but who refer to themselves as "Naáyarite".

 The Cora inhabit the northern sierra of the Mexican state Nayarit which is named after its indigenous inhabitants.

 Cora is a Mesoamerican language and shows many of the traits defining the Mesoamerican Linguistic Area.

 Under the General Law of Linguistic Rights of the Indigenous Peoples it is recognized as a "national language" along with 62 other indigenous languages and
Spanish which have the same "validity" in Mexico .
</item 2>

<item 3>
Achomawi
In anthropology, Achomawi (also Achumawi, Ajumawi and Ahjumawi) are the northerly nine (out of eleven) bands of the Pit River tribe of Native Americans who
lived in what is now northeastern California in the United States.

 These nine autonomous bands (also called "tribelets") of the Pit River Indians spoke various dialects of one common language, and the other two bands spoke
dialects of a related language, called Atsugewi.

 The name "Achomawi" means river dwelling (from ajumma, "river") and the band historically inhabited the Fall River Valley and the Pit River from the south end
of Big Valley Mountains, westerly to Pit River Falls.
</item 3>

<item 4>
Pagode (music)
Pagode is a type of Brazilian country-folk traditional style of music.

 It's also known as Cipó Preto (black liana), Pagode caipira (rural pagode) or Pagode sertanejo (folk pagode).

 Sertanejo means anything that comes from the back-country, outback or simply the countryside.

 This style of music was pioneered by musicians Tião Carreiro and Lourival dos Santos in the late 1950s, when they fused the "Coco" and "Calango de roda"
rhythms.

 Although other instruments can be added, it is much characterized by its simple, acoustic approach, with the use of just one or two viola guitars, often
accompanied by an acoustic guitar and, at times, a light percussion section.

 A structural feature commonly used in this genre is the alternation of verses sung with the accompaniement of a classical guitar, with solos of viola caipira.

 Pieces can be entirely instrumental or have one or two singers, usually: when there are two lead vocals (that can be men, women or a combination), they sing
together on a steady interval.
</item 4>

<item 5>
Guarani language
Guarani ( or ), specifically the primary variety known as Paraguayan Guarani (endonym "avañe'ẽ" ] 'the people's language'), is an indigenous language of South
America that belongs to the Tupi–Guarani family of the Tupian languages.

 It is one of the official languages of Paraguay (along with Spanish), where it is spoken by the majority of the population, and where half of the rural
population is monolingual.

 It is spoken by communities in neighboring countries, including parts of northeastern Argentina, southeastern Bolivia and southwestern Brazil, and is a second
official language of the Argentine province of Corrientes since 2004; it is also an official language of Mercosur.
</item 5>

<item 6>
Indigenous language
An indigenous language or autochthonous language is a language that is native to a region and spoken by indigenous people, often reduced to the status of a
minority language.

 This language would be from a linguistically distinct community that has been settled in the area for many generations.

 Indigenous languages are not necessarily national languages, and the reverse is also true.
</item 6>

<item 7>
Catuaba
The name catuaba (pronounced [ka.two.

'aba], a Guarani word that means "what gives strength to the Indian") is used for the infusions of the bark of a number of trees native to Brazil.

 The most widely used barks are derived from the trees "Trichilia catigua" and "Erythroxylum vaccinifolium".

 Other catuaba preparations use the bark of trees from the following genera or families: "Anemopaegma", "Ilex", "Micropholis", "Phyllanthus", "Secondatia",
"Tetragastris" and species from the Myrtaceae.
</item 7>

<item 8>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 8>

<item 9>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 9>

<item 10>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 10>
=== END ITEM id=ce8c01fed92fc4b5 ===

=== ITEM id=2fb4a4e76eccc91e ===
Gold sentences and facts:
Sentence 0:  The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
  Fact 0.0: The show originally ran from 7 September 1977 to 20 November 1995.
  Fact 0.1: The show was hosted by Gordon Burns.
  Fact 0.2: The show was usually broadcast on the ITV network.
  Fact 0.3: The show was usually broadcast on Mondays at 7pm.
Sentence 1:  He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West Tonight" from January 1997 to October 2011.
  Fact 1.0: The person referred to as 'He' was the host of 'The Krypton Factor' for its original 18-year run.
  Fact 1.1: The Krypton Factor's original run was 1977–1995.
  Fact 1.2: The person referred to as 'He' was the chief anchorman for the BBC regional news programme 'North West Tonight'.
  Fact 1.3: The person referred to as 'He' was the chief anchorman for 'North West Tonight' from January 1997 to October 2011.

Retrieved items:
<item 1>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 1>

<item 2>
The Krypton Factor
The Krypton Factor is a British game show produced by Granada for broadcast on ITV.

 The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
</item 2>

<item 3>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 3>

<item 4>
Night After Night with Allan Havey
Night After Night with Allan Havey was a late night talk show which aired on The Comedy Channel and, later, Comedy Central from 1989 to 1992.

 The show featured the stream of consciousness of host Allan Havey, and other departures from the standard late-night talk show format.

 Havey's sidekick was Nick Bakay, who moved on to act as sidekick on Dennis Miller's short-lived late night talk show.
</item 4>

<item 5>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 5>

<item 6>
Gordon Burns (footballer)
Gordon Burns (born 2 December 1978) is a Scottish former professional footballer who is currently manager of Troon in the Scottish Junior Football Association,
West Region.

 He has previously played in the Scottish Championship for Ayr United.
</item 6>

<item 7>
At Your Request
At Your Request was an Australian television daytime series which aired from 1958 to 1959.

 The series aired on Tuesdays at 2:30PM on Melbourne station HSV-7, and was hosted by baritone Charles Skase, who was also known as a radio personality.

 Information on this series is scarce.

 The series presented requested songs, but it is not clear how these songs were presented (such as whether it was a disc jockey series like "TV Disc Jockey", a
lip-sync series like "Hit Parade", or a live music series like "Sweet and Low").

 The archival status of the series is also not known, although being a daytime series aired in a single city means it is unlikely (though not impossible) that
kinescope recordings exist of it.
</item 7>

<item 8>
Gordon Burns
Gordon Henry Burns (born 10 June 1942) is a Northern Irish journalist and broadcaster.

 He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West
Tonight" from January 1997 to October 2011.

 In November 2011 Gordon moved back to Belfast were he was born.
</item 8>

<item 9>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 9>

<item 10>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 10>
=== END ITEM id=2fb4a4e76eccc91e ===

=== ITEM id=a354fe439557ae49 ===
Gold sentences and facts:
Sentence 0: Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.
  Fact 0.0: Lake Louisvilla is a neighborhood.
  Fact 0.1: Lake Louisvilla is partially located in Louisville, Kentucky.
Sentence 1:  It is located between Westport Road in Louisville and KY 22 in Oldham County.
  Fact 1.0: Lake Louisvilla is located between Westport Road in Louisville.
  Fact 1.1: Lake Louisvilla is located between KY 22 in Oldham County.
Sentence 2: Oldham County is a county located in the commonwealth of Kentucky.
  Fact 2.0: Oldham County is a county.
  Fact 2.1: Oldham County is located in the commonwealth of Kentucky.
Sentence 3:  As of the 2010 census, the population was 60,316.
  Fact 3.0: As of the 2010 census, the population of Oldham County was 60,316.
Sentence 4:  Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.
  Fact 4.0: Oldham County was a prohibition or completely dry county until January 2005.
  Fact 4.1: Oldham County stopped being completely dry in January 2005 as the result of a 2004 'moist' vote.
  Fact 4.2: The 2004 'moist' vote permitted sales of alcohol in restaurants that seat at least 100 patrons.
  Fact 4.3: The 2004 'moist' vote permitted alcohol sales only in restaurants where 70% or more of total revenue is derived from sales of food.

Retrieved items:
<item 1>
Oldham County, Kentucky
Oldham County is a county located in the commonwealth of Kentucky.

 As of the 2010 census, the population was 60,316.

 Its county seat is La Grange.

 The county is named for Colonel William Oldham.

 Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants
that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.

 After a vote in late 2015; Oldham county has become a completely wet county.
</item 1>

<item 2>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 2>

<item 3>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 3>

<item 4>
Garrard County, Kentucky
Garrard County ( ;) is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 16,912.

 Its county seat is Lancaster.

 The county was formed in 1796 and was named for James Garrard, Governor of Kentucky from 1796 to 1804.

 It is a prohibition or dry county but Lancaster is wet.

 Lancaster was founded as a settlement of log cabins in 1776 at a springs that later provided a constant source of water to early pioneers.

 It is one of the oldest cities in the Commonwealth.

 Boonesborough, 25 miles to the east, was founded by Daniel Boone in 1775.

 Lexington, 28 miles to the north, was founded in 1775.

 Stanford, originally known as St. Asaph, is 10 miles south of Lancaster.

 It too was founded in 1775.

 The oldest permanent settlemenet in Kentucky, Harrodsburg, was founded in 1774 and is 18 miles to the west.

 The present day courthouse is one of the oldest courthouses in Kentucky in continuous use.
</item 4>

<item 5>
KFC
KFC, until 1991 known as Kentucky Fried Chicken, is an American fast food restaurant chain that specializes in fried chicken.

 Headquartered in Louisville, Kentucky, it is the world's second-largest restaurant chain (as measured by sales) after McDonald's, with almost 20,000 locations
globally in 123 countries and territories as of December 2015 .

 The chain is a subsidiary of Yum!

 Brands, a restaurant company that also owns the Pizza Hut and Taco Bell chains.
</item 5>

<item 6>
Kentucky County, Virginia
Kentucky County (then alternately spelled Kentucke County) was formed by the Commonwealth of Virginia by dividing Fincastle County into three new counties:
Kentucky, Washington, and Montgomery, effective December 31, 1776.

 During the three and one-half years of Kentucky County's existence, its seat of government was Harrodstown (then also known as Oldtown, later renamed
Harrodsburg).
</item 6>

<item 7>
Nashville Terminal Subdivision
There are two crossings near each other, Williams Ave and Nesbitt Lane at Amqui.

 From here the Terminal goes south about 2 miles to the Nashville National Cemetery to the first major location, known as Ekin, 000179, where there is a cross
over track from number 1 to number 2 track (left to right track).

 There is also the first EDD (Defect Detector) at 000179.1.

 Few more miles southward, the next major location appears.

 At 000181.0, Maplewood is a major location for the Terminal.

 From here the regular Terminal goes south to swap crews at Kayne Ave, and the right side, Radnor Cutoff, detours the city and gives yard departures and
arrivals direct access to and from Radnor yard to cities like Louisville and Chicago.

 The Cutoff runs from Maplewood to Shelby Park double track.

 From Shebly the track converges into one to pass the historic Shelby bridge, then it opens back to double track.

 From there, the cutoff hits the Intermediates at 0BA187.0 known as Chicken Pike.
</item 7>

<item 8>
Lake Louisvilla, Louisville
Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.

 It is located between Westport Road in Louisville and KY 22 in Oldham County.

 Lake Louisvilla was developed in the 1920s as a summer resort for people living in the city of Louisville.

 The state of Kentucky drained the lake in the late 1980s due to safety concerns regarding the stability of a dam.
</item 8>

<item 9>
Westervelt massacre
The Westervelt massacre, also known as the Westerfield massacre, was an indigenous attack on a caravan of Dutch frontier settlers that occurred during the
American Revolutionary War around 3:00 am on June 27, 1780 in Kentucky County, Virginia, the present day state of Kentucky.

 It remains one of the largest massacres in Kentucky state history.

 The settlers were traveling southeast from Low Dutch Station to Harrod's Town.

 The settler relocation was in part a reaction to British Captain Henry Bird's invasion of Kentucky.

 The area immediately east of Low Dutch Station had been overrun with British allied Indians.

 Harrod's Town was fortified and a move south would lead the settlers away from Captain Bird's invading army from the north.

 The caravan was ambushed in a surprise attack, during the night, after a day's travel of twelve miles.

 The exact location of the massacre is not definitively known.

 However, it is likely to have occurred at Floyd's Fork and Broad Run.

 The caravan was formed by Jacobus Westervelt and consisted of forty-one settlers from ten different families.

 Ten of the seventeen settlers killed were members of the Westervelt family.
</item 9>

<item 10>
Casey County, Kentucky
Casey County is a county located in the U.S. Commonwealth of Kentucky.

 As of the 2010 census, the population was 15,955.

 Its county seat is Liberty.

 The county was formed in 1806 from the western part of Lincoln County and named for Colonel William Casey, a pioneer settler who moved his family to Kentucky
in 1779.

 It is the only Kentucky county entirely in the Knobs region.

 Casey County is home to annual Casey County Apple Festival, and is a prohibition or dry county.

 It is considered part of the Appalachian region of Kentucky.
</item 10>
=== END ITEM id=a354fe439557ae49 ===

=== ITEM id=090f3b8e157c1d18 ===
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
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 4>

<item 5>
Valentina (hot sauce)
Valentina is a brand of pourable hot sauce manufactured by Salsa Tamazula, a company in Guadalajara, Mexico.

 It is typically sold in 12.5-ounce and very large (one-liter or 34-ounce) glass bottles, with a flip-top cap permanently attached to the bottle (the cap does
not unscrew).

 The sauce, and the parent company's Tamazula hot sauce, are made with puya chilis from Jalisco state – similar to the Guajillo chili and known by the name
"guajillo puya" – is described as thicker than Tabasco sauce and less vinegary, with more chile flavor.

 Valentina comes in two varieties, hot and extra hot, and the sauce is known for its use on tacos and its taste, as opposed to only for its heat.

 Valentina's ingredients are water, chili peppers, vinegar, salt, spices and sodium benzoate (as a preservative).
</item 5>

<item 6>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
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
Riverton, New York was a planned community built in 1973 in the town of Henrietta, New York, along the Genesee River.

 Riverton was seventh of the thirteen communities receiving Federal guarantee assistance from HUD as part of their New Communities Program created by the Urban
Growth and New Community Development Act of 1970.

 $12 million was earmarked for the community.

 Riverton was supposed to encompass a 2335 acre land area bordered on the north by the New York State Thruway, on the west by the Genesee River, on the east by
East River Rd., and on the south by the Rush-Henrietta town line, with some development continuing east along Erie Station Rd.

 Original plans called for construction to occur for a 16-year period and eventually house over 25,000 people, with the community including an 18-hole golf
course, 12 swimming pools, 3 artificial lakes, a marina on the Genesee, and commercial and industrial centers.

 However, only a small portion of this was built along Scottsville-West Henrietta Rd.
</item 8>

<item 9>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 9>

<item 10>
Sriracha sauce
Sriracha (Thai: ศรีราชา ,  ] ; ) is a type of hot sauce or chili sauce made from a paste of chili peppers, distilled vinegar, garlic, sugar, and salt.

 It is named after the coastal city of Si Racha, in Chonburi Province of eastern Thailand, where it may have been first produced for dishes served at local
seafood restaurants.
</item 10>
=== END ITEM id=090f3b8e157c1d18 ===

=== ITEM id=4568a75e7a1ff908 ===
Gold sentences and facts:
Sentence 0:  The event was broadcast live on BBC One and was hosted by Graham Norton.
  Fact 0.0: The event was broadcast live on BBC One.
  Fact 0.1: Graham Norton hosted the event.
Sentence 1: Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.
  Fact 1.0: Graham William Walker was born on 4 April 1963.
  Fact 1.1: Graham William Walker is better known as Graham Norton.
  Fact 1.2: Graham Norton is an Irish television presenter.
  Fact 1.3: Graham Norton is an Irish radio presenter.
  Fact 1.4: Graham Norton is an Irish DJ.
  Fact 1.5: Graham Norton is an Irish comedian.
  Fact 1.6: Graham Norton is an Irish actor.
  Fact 1.7: Graham Norton is an Irish writer.

Retrieved items:
<item 1>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 1>

<item 2>
British Academy Children's Awards
The British Academy Children's Awards are presented in an annual award show hosted by the British Academy of Film and Television Arts (BAFTA).

 They have been awarded annually since 1996.

 Before that, children's awards were a part of the main British Academy Television Awards.
</item 2>

<item 3>
Graham Norton
Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.

 He is a five-time BAFTA TV Award winner for his comedy chat show "The Graham Norton Show".

 Previously shown on BBC Two, it took the prestigious Friday night slot on BBC One from "The Jonathan Ross Show" in 2010.
</item 3>

<item 4>
BAFTA Scotland
BAFTA in Scotland is the Scottish branch of the British Academy of Film and Television Arts.

 Formed in 1986, the branch holds two annual awards ceremonies recognising the achievement by performers and production staff in Scottish film, television and
video games.

 These Awards are separate from the British Academy Television Awards and British Academy Film Awards.
</item 4>

<item 5>
British Academy Television Awards
The British Academy Television Awards, also known as the BAFTA Television Awards, are presented in an annual award show hosted by the British Academy of Film
and Television Arts (BAFTA).

 They have been awarded annually since 1955.

 It is the British equivalent to the Emmy Awards in the United States and Logie Awards in Australia.
</item 5>

<item 6>
2012 British Academy Television Awards
The 2012 British Academy Television Awards (formally known as the Arqiva British Academy Television Awards) were held on 27 May 2012 at the Royal Festival Hall
in London.
</item 6>

<item 7>
British Academy Television Awards 1997
The 1997 British Academy Television Awards were held on 29 April at the Royal Albert Hall in London, as a joint ceremony with the British Academy Film Awards.

 To date, it is the last occasion upon which the two sets awards have been given jointly at the same event; from 1998, the Film and Television Awards were given
at separate ceremonies, for the first time since 1968.
</item 7>

<item 8>
2013 Africa Magic Viewers Choice Awards
The 2013 Africa Magic Viewers Choice Awards were held on 9 March 2014 and co-hosted by Big Brother Africa presenter IK Osakioduwa and StarGist’s host, Vimbai
Mutinhiri.

 Ivie Okujaiye and Olu Jacobs were honored with the TrailBlazer and Industry Merit Awards respectively.
</item 8>

<item 9>
British Academy Television Award for Best Actress
This is a list of the British Academy Television Awards for Best Actress.

 The British Academy Television Awards began in 1955.

 The Best Actress award was initially given as an 'individual honour' without credit to a particular performance until 1969 when Wendy Craig won for her
performance in "Not in Front of the Children".

 Since 1970, nominees have been announced in addition to the winner and are listed with the winner highlighted in blue.

 The Actress category was split into Leading Actress and Supporting Actress starting in 2010.
</item 9>

<item 10>
British Academy Television Awards 1998
The 1998 British Academy Television Awards were held on 18 May at the Prince of Wales Theatre in London.

 The ceremony was broadcast on ITV, hosted by Bob Monkhouse, and it was the first occasion since 1968 that the Television Awards had been held separately from
the British Academy Film Awards, instead of as a joint ceremony.
</item 10>
=== END ITEM id=4568a75e7a1ff908 ===

=== ITEM id=f45a7591bb904339 ===
Gold sentences and facts:
Sentence 0:  The event was broadcast live on BBC One and was hosted by Graham Norton.
  Fact 0.0: The event was broadcast live on BBC One.
  Fact 0.1: Graham Norton hosted the event.
Sentence 1: Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.
  Fact 1.0: Graham William Walker was born on 4 April 1963.
  Fact 1.1: Graham William Walker is better known as Graham Norton.
  Fact 1.2: Graham Norton is an Irish television presenter.
  Fact 1.3: Graham Norton is an Irish radio presenter.
  Fact 1.4: Graham Norton is an Irish DJ.
  Fact 1.5: Graham Norton is an Irish comedian.
  Fact 1.6: Graham Norton is an Irish actor.
  Fact 1.7: Graham Norton is an Irish writer.

Retrieved items:
<item 1>
Graham Norton
Graham William Walker (born 4 April 1963), better known as Graham Norton, is an Irish television and radio presenter, DJ, comedian, actor, and writer.

 He is a five-time BAFTA TV Award winner for his comedy chat show "The Graham Norton Show".

 Previously shown on BBC Two, it took the prestigious Friday night slot on BBC One from "The Jonathan Ross Show" in 2010.
</item 1>

<item 2>
British Academy Television Awards 1997
The 1997 British Academy Television Awards were held on 29 April at the Royal Albert Hall in London, as a joint ceremony with the British Academy Film Awards.

 To date, it is the last occasion upon which the two sets awards have been given jointly at the same event; from 1998, the Film and Television Awards were given
at separate ceremonies, for the first time since 1968.
</item 2>

<item 3>
Graph query row:
person: Graham Norton
date_of_birth: 4 April 1963
</item 3>

<item 4>
British Academy Television Award for Best Actress
This is a list of the British Academy Television Awards for Best Actress.

 The British Academy Television Awards began in 1955.

 The Best Actress award was initially given as an 'individual honour' without credit to a particular performance until 1969 when Wendy Craig won for her
performance in "Not in Front of the Children".

 Since 1970, nominees have been announced in addition to the winner and are listed with the winner highlighted in blue.

 The Actress category was split into Leading Actress and Supporting Actress starting in 2010.
</item 4>

<item 5>
British Academy Television Awards
The British Academy Television Awards, also known as the BAFTA Television Awards, are presented in an annual award show hosted by the British Academy of Film
and Television Arts (BAFTA).

 They have been awarded annually since 1955.

 It is the British equivalent to the Emmy Awards in the United States and Logie Awards in Australia.
</item 5>

<item 6>
British Academy Children's Awards
The British Academy Children's Awards are presented in an annual award show hosted by the British Academy of Film and Television Arts (BAFTA).

 They have been awarded annually since 1996.

 Before that, children's awards were a part of the main British Academy Television Awards.
</item 6>

<item 7>
British Academy Television Awards 1998
The 1998 British Academy Television Awards were held on 18 May at the Prince of Wales Theatre in London.

 The ceremony was broadcast on ITV, hosted by Bob Monkhouse, and it was the first occasion since 1968 that the Television Awards had been held separately from
the British Academy Film Awards, instead of as a joint ceremony.
</item 7>

<item 8>
BAFTA Scotland
BAFTA in Scotland is the Scottish branch of the British Academy of Film and Television Arts.

 Formed in 1986, the branch holds two annual awards ceremonies recognising the achievement by performers and production staff in Scottish film, television and
video games.

 These Awards are separate from the British Academy Television Awards and British Academy Film Awards.
</item 8>

<item 9>
2012 British Academy Television Awards
The 2012 British Academy Television Awards (formally known as the Arqiva British Academy Television Awards) were held on 27 May 2012 at the Royal Festival Hall
in London.
</item 9>

<item 10>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 10>
=== END ITEM id=f45a7591bb904339 ===

=== ITEM id=d16bbbe8145a7ae0 ===
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
10,000 metres
The 10,000 metres or 10,000-meter run is a common long-distance track running event.

 The event is part of the athletics programme at the Olympic Games and the World Championships in Athletics and is common at championship level events.

 The race consists of 25 laps around an Olympic-sized track.

 It is less commonly held at track and field meetings, due to its duration.

 The 10,000 metres track race is usually distinguished from its road running counterpart, the 10K run, by its reference to the distance in metres rather than
kilometres.
</item 1>

<item 2>
Harry Prowell
Harry Prowell A.A.(10 July 1936 – 27 June 2000) was a Guyanese long distance runner who represented Guyana in the Marathon at the 1968 Summer Olympics in Mexico
City, Mexico.

 He is known to be one of the greatest Marathon runners Guyana has ever produced, setting the national record in 1968.

 To date, he is the only Guyanese ever to compete in the Marathon at the Summer Olympic Games and one of the most prominent Indo-Caribbean long distance runners
of his time.

 He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth
Games in Edinburgh, Scotland.
</item 2>

<item 3>
Dave Bailey (athlete)
David Bailey (born March 17, 1945 in Toronto, Ontario) is a retired track and field athlete, who represented Canada at the 1968 Summer Olympics in the men's
1.500 metres.

 He was the first Canadian to run the mile in less than 4 minutes (3:59.1) in San Diego, CA on June 11, 1966 and the first Canadian to the run the mile in less
than 4 minutes in Canada (3:57.7)in Toronto on July 22, 1967.

 A resident of Willowdale, Ontario he won the bronze medal in this event at the 1967 Pan American Games and the silver medal in this event at the 1968 World
University Games He was two-time Canadian Universities Track and Field Athlete of Year (1965, 1967) and two-time inductee into the University of Toronto Sports
Hall of Fame (individually 1998 and team 2003).
</item 3>

<item 4>
Dave Bailey (athlete)
He was a member of 9 Canadian National Track and Field Teams competing at World University Games, Budapest 1965 and Tokyo 1967, Commonwealth Games, Kingston,
Jamaica, 1966, Pan American Games, Winnipeg, 1967, Commonwealth vs USA, Los Angeles, 1967, Pre-Olympic Games, Mexico City, 1967, Soviet Union Tour, 1968,
European Tour, 1968, Olympic Games, Mexico City, 1968.
</item 4>

<item 5>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 5>

<item 6>
Ross MacKenzie (athlete)
Ross MacKenzie (born 18 July 1946) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1968 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Brian MacLaren, Bill Crothers, and Robert McLaren).

 MacKenzie also finished fifth in the 1967 Pan American Games 400 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Brian MacLaren and Bill Crothers.
</item 6>

<item 7>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 7>

<item 8>
2011 Pan American Games
The 2011 Pan American Games, officially the XVI Pan American Games, was an international multi-sport event that was held from October 14–30, 2011, in
Guadalajara, Jalisco, Mexico.

 Some events were held in the nearby cities of Ciudad Guzmán, Puerto Vallarta, Lagos de Moreno and Tapalpa.

 It was the largest multi-sport event of 2011, with approximately 6,000 athletes from 42 nations participating in 36 sports.

 Both the Pan American and Parapan American Games were organized by the Guadalajara 2011 Organizing Committee (COPAG).

 The 2011 Pan American Games were the third Pan American Games hosted by Mexico (the first country to do so) and the first held in the state of Jalisco.

 Previously, Mexico hosted the 1955 Pan American Games and the 1975 Pan American Games, both in Mexico City.

 The 2011 Parapan American Games were held 20 days after the Pan American Games have ended.
</item 8>

<item 9>
Nora Rocha
Nora Leticia Rocha de la Cruz (born December 18, 1967 in Monclova, Coahuila) is a retired female track and field athlete from Mexico, who competed in the 5000
and 10,000 metres.

 She claimed the gold medal in the women's 10,000 metres at the 1999 Pan American Games in Winnipeg, Canada.
</item 9>

<item 10>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 10>
=== END ITEM id=d16bbbe8145a7ae0 ===

=== ITEM id=1b733035dc275bc6 ===
Gold sentences and facts:
Sentence 0: Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.
  Fact 0.0: Lowell J. Sherman was born on October 11, 1885.
  Fact 0.1: Lowell J. Sherman died on December 28, 1934.
  Fact 0.2: Lowell J. Sherman was an American actor.
  Fact 0.3: Lowell J. Sherman was an American film director.
Sentence 1: Jonathan Kaplan (born November 25, 1947) is an American film producer and director.
  Fact 1.0: Jonathan Kaplan was born on November 25, 1947.
  Fact 1.1: Jonathan Kaplan is an American film producer.
  Fact 1.2: Jonathan Kaplan is an American film director.

Retrieved items:
<item 1>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 1>

<item 2>
Lowell Sherman
Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.

 At a time when it was highly unusual, he was both the actor and director on several films in the early 1930s, before completely transitioning to the role of
director.

 At the height of his career, after scoring huge successes with his direction of the films "She Done Him Wrong" and "Morning Glory" (which starred Mae West and
won her first Academy Award for Katharine Hepburn respectively), he succumbed to pneumonia after a brief illness.
</item 2>

<item 3>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 3>

<item 4>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 4>

<item 5>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 5>

<item 6>
Jonathan Kaplan
Jonathan Kaplan (born November 25, 1947) is an American film producer and director.

 His film "The Accused" (1988) earned actress Jodie Foster her first Oscar for Best Actress and was nominated for the Golden Bear at the 39th Berlin
International Film Festival.

 Kaplan received five Emmy nominations for his roles directing and producing "ER".
</item 6>

<item 7>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 7>

<item 8>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 8>

<item 9>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 9>

<item 10>
Maurice Browne
Maurice Browne (12 February 1881 – 21 January 1955), born in Reading, England, was best known as a theater producer in the United States and the UK.

 The Cambridge-educated Browne was also a poet, actor, and theater director.

 He has been credited, along with his then-wife Ellen Van Volkenburg, with being the founder of the Little Theatre Movement in America through his work with the
Chicago Little Theatre.

 Browne and Van Volkenburg went on to found the department of drama at the Cornish School in Seattle in 1918, now Cornish College of the Arts.

 Browne's greatest triumph came in 1929 when he produced "Journey's End", by R. C. Sherriff in London.
</item 10>
=== END ITEM id=1b733035dc275bc6 ===

=== ITEM id=3ebaa24ee53e3eae ===
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
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 1>

<item 2>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
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
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
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
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
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
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 8>

<item 9>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 9>

<item 10>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 10>
=== END ITEM id=3ebaa24ee53e3eae ===
