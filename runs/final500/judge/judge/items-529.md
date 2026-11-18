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

=== ITEM id=a595f0dd151bd2db ===
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
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 1>

<item 2>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 2>

<item 3>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 3>

<item 4>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 4>

<item 5>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 5>

<item 6>
Bosko the Musketeer
Bosko the Musketeer is an American animated short film.

 It is a "Looney Tunes" cartoon, featuring Bosko, the first star of the series.

 It was released on August 12, 1933, although some sources note September 16 as a date; this is problematic, as that would imply that the last films featuring
Bosko as the star of Warner Bros. cartoons were released after the first film featuring Buddy, the second star of the series.

 It was, like most "Looney Tunes" of the time, directed by Hugh Harman; Frank Marsales was the musical director.
</item 6>

<item 7>
Bambi
Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi,
a Life in the Woods" by Austrian author Felix Salten.

 The film was released by RKO Radio Pictures on August 13, 1942, and is the fifth Disney animated feature film.
</item 7>

<item 8>
Get a Horse!
Get a Horse!

 is a 2013 American 3D animated slapstick comedy short film, produced by Walt Disney Animation Studios.

 Combining black-and-white hand-drawn animation and color CGI animation, the short features the characters of the late 1920s "Mickey Mouse" cartoons, and
features archival recordings of Walt Disney in his posthumous role as Mickey Mouse.

 It is the first original Mickey Mouse theatrical animated short since "Runaway Brain" (1995), and the first appearance of Oswald the Lucky Rabbit in a Disney
animated production in 85 years.
</item 8>

<item 9>
Walt Disney Pictures, founded under that name in 1983, produces live-action films; Walt Disney Studios Motion Pictures distributes Disney studio releases,
succeeding Buena Vista (1953–2007). Disney highlights include *Bambi* (1942), *Pinocchio* (1940), and the 1989–99 Renaissance, spanning ten animated films from
*The Little Mermaid* to *Tarzan*. Buddy Baker scored several Disney films. Other entries cover Mickey and Donald, the
</item 9>

<item 10>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 10>
=== END ITEM id=a595f0dd151bd2db ===

=== ITEM id=435061ee14b00e1e ===
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
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 1>

<item 2>
FSC Star
Fabryka Samochodów Ciężarowych "Star" was a Polish truck manufacturer.

 The name comes from the City of Starachowice, where the factory is located.

 Their first vehicle was the "Star 20" in 1948.

 The most popular product was the "Star 266".

 The 266 model offered very good quality and powerful engines for a low price.

 It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).

 For many years FSC Star was a state-owned company.

 Star is now owned by MAN AG who eliminated the brand in January 2009.
</item 2>

<item 3>
Leader Trucks
Leader found their niche in the heavy transport market and built as a rigid 4x4 and 6x6 configuration for tray, tipper and agitator applications, they offered
reliable machines in the industries of earth-moving and off-road construction.

 Leader was notable for being the first manufacturer to fit Caterpillar engines to trucks, to fit automatic transmissions to diesel trucks in assembly, and the
first truck manufacturer in Australia to offer disc brakes.

 Leader boasted its components to be 80 percent Australian built with only the engine, transmission and steering box imported.

 By 1980, Leader had sold its 1000th truck in Australia and had also sold trucks to New Zealand, South Africa, Hong Kong, China and Indonesia.
</item 3>

<item 4>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 4>

<item 5>
E-Z Polish Factory
The E-Z Polish Factory is a building designed by Frank Lloyd Wright and is located in Chicago, Illinois, United States.

 Built in 1905, the E-Z Polish Company made polish for shoes and stoves.

 It now serves as practice space for local bands and artists.
</item 5>

<item 6>
Volvo FH
The Volvo FH is a heavy truck range produced by Swedish truck manufacturer Volvo Trucks.

 Introduced in late 1993 as FH12 and FH16, production still continues with the now the second generation of FH range model lineup.

 FH stands for Forward control High entry where numbers denominate engine capacity in litres.

 The FH range is one of the most successful truck series ever having sold more than 400,000 units worldwide.
</item 6>

<item 7>
Shaanxi Automobile Group
Shaanxi Automobile Group Co., Ltd. is a Chinese bus and truck manufacturer headquartered in Xi'an, Shaanxi.

 It employs around 23,000 employees.

 It manufactures heavy duty truck, buses (chassis), medium-size heavy truck and heavy duty axles for trucks.

 It utilizes Magna Steyr and MAN SE technologies.

 Its bus chassis are sold under the Eurostar Bus brand.
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
J. A. Baczewski
J. A. Baczewski was a Polish company most well known for its fine spirits such as vodka and gin.

 The factory, dating back to the late 18th century, was based in Lwów (Lviv) and until 1939 was one of two most popular Polish export goods.
</item 9>

<item 10>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 10>
=== END ITEM id=435061ee14b00e1e ===

=== ITEM id=9baab296d9bcb1c7 ===
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
Contraband (2012 film)
Contraband is a 2012 American crime thriller film directed by Baltasar Kormákur, starring Mark Wahlberg, Kate Beckinsale, Ben Foster, Caleb Landry Jones,
Giovanni Ribisi, Lukas Haas, Diego Luna and J. K. Simmons.

 The film is a remake of the 2008 Icelandic film "Reykjavík-Rotterdam" which Baltasar Kormákur starred in.

 It was released on January 13, 2012 in the United States by Universal Pictures.
</item 1>

<item 2>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 2>

<item 3>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 3>

<item 4>
Deadfall (2012 film)
Deadfall is a 2012 American crime drama film directed by Stefan Ruzowitzky, written by Zach Dean and starring Eric Bana, Olivia Wilde, and Charlie Hunnam.
</item 4>

<item 5>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 5>

<item 6>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 6>

<item 7>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 7>

<item 8>
Hercules, Prisoner of Evil
Hercules, Prisoner of Evil (Italian: "Ursus il terrore dei Kirghisi" , 'Ursus, Terror of the Kirghiz' ) is a 1964 Italian "peplum" film directed by Antonio
Margheriti and an uncredited Ruggero Deodato.

 Deodato, the official assistant director, replaced Margheriti as he was busy with the completion of the film "The Fall of Rome".

 Deodato actually directed most of the film in actuality but Margheriti was credited as the director.

 The film is filled with a variety of horrific themes and elements, featuring a killer werewolf, and is as much a horror film as it is a peplum.
</item 8>

<item 9>
Milan (2004 film)
Milan is a 2004 romantic drama film released under Star Cinema, ABS-CBN Film Productions, Inc. in the Philippines.

 It stars Claudine Barretto and Piolo Pascual.

 It is a love story written by Raymond Lee, and under Olivia Lamasan's direction.

 The movie made more than P100 million in the box office.
</item 9>

<item 10>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 10>
=== END ITEM id=9baab296d9bcb1c7 ===

=== ITEM id=c14e95e8659b9ff9 ===
Gold sentences and facts:
Sentence 0: Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
  Fact 0.0: Gunnar Eric Nelson was born on September 20, 1967.
  Fact 0.1: Gunnar Eric Nelson is an American musician.
  Fact 0.2: Gunnar Eric Nelson is an American singer.
  Fact 0.3: Gunnar Eric Nelson is an American songwriter.
  Fact 0.4: Gunnar Eric Nelson is an international multi-platinum recording artist.
Sentence 1: Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the Machine.
  Fact 1.0: Florence Leontine Mary Welch was born on 28 August 1986.
  Fact 1.1: Florence Leontine Mary Welch is an English musician.
  Fact 1.2: Florence Leontine Mary Welch is an English singer.
  Fact 1.3: Florence Leontine Mary Welch is an English songwriter.
  Fact 1.4: Florence Leontine Mary Welch is best known as the vocalist of the indie rock band Florence and the Machine.

Retrieved items:
<item 1>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 1>

<item 2>
Gwendolyn Elaine Armstrong
Gwendolyn Elaine Armstrong was a black Mississippi pioneer in the Civil Rights Movement.

 In September, 1965, she and Raylawni Branch, both local natives, integrated the University of Southern Mississippi at Hattiesburg.

 They thus completed the process of breaking the segregation barriers at Mississippi’s universities which had been begun by Clyde Kennard at (then) Mississippi
Southern College (1956–61) and carried forward by James Meredith at the University of Mississippi (September, 1962) and Richard Holmes at Mississippi State
University (July, 1965).
</item 2>

<item 3>
Nelson (band)
Nelson is an American rock band founded by singer/songwriters Matthew and Gunnar Nelson (twin sons of Ricky Nelson and Kristin Nelson).

 The band achieved success during the early 1990s with their double platinum debut album "After the Rain", which featured the number-one hit "(Can't Live
Without Your) Love and Affection".
</item 3>

<item 4>
Gunnar Nelson (musician)
Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
</item 4>

<item 5>
What the Water Gave Me (song)
"What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.

 It was released on 23 August 2011 as the first promotional single from "Ceremonials".

 After the band performed the song in Berkeley, California on 12 June 2011, the studio version premiered on Florence and the Machine's official website.

 Welch decided to name the song "What the Water Gave Me" after seeing a Frida Kahlo painting of the same name.

 During an interview, she confirmed that the song was inspired by English writer Virginia Woolf.
</item 5>

<item 6>
Sweet Nothing
"Sweet Nothing" is a song by Scottish recording artist and record producer Calvin Harris from his third studio album, "18 Months" (2012).

 It features vocals from Florence and the Machine singer Florence Welch.

 Harris previously worked with the band in a remix of their single, "Spectrum" (2012).

 While recording "18 Months", Harris expressed interest in working with Welch.

 He mentioned that it was not easy and he had to chase her, due to schedule conflicts.

 Welch accepted his invitation and the two recorded "Sweet Nothing".

 The track premiered on "The Chris Moyles Show" on BBC Radio 1 on 28 August 2012 and was later released as the fifth single from the album on 12 October 2012.
</item 6>

<item 7>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
</item 7>

<item 8>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 8>

<item 9>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 9>

<item 10>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 10>
=== END ITEM id=c14e95e8659b9ff9 ===

=== ITEM id=7ca8f0b138dc46b0 ===
Gold sentences and facts:
Sentence 0: Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.
  Fact 0.0: Cars 2 is a film.
  Fact 0.1: Cars 2 was released in 2011.
  Fact 0.2: Cars 2 is American.
  Fact 0.3: Cars 2 is computer-animated.
  Fact 0.4: Cars 2 is an action comedy spy film.
  Fact 0.5: Cars 2 was produced by Pixar Animation Studios.
  Fact 0.6: Cars 2 was produced for Walt Disney Pictures.
Sentence 1: Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway musicals "Hamilton" and "In the Heights".
  Fact 1.0: Lin-Manuel Miranda was born January 16, 1980.
  Fact 1.1: Lin-Manuel Miranda is American.
  Fact 1.2: Lin-Manuel Miranda is a composer, lyricist, playwright, and actor.
  Fact 1.3: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical Hamilton.
  Fact 1.4: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical In the Heights.
Sentence 2:  He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".
  Fact 2.0: Lin-Manuel Miranda co-wrote the songs for Disney's Moana soundtrack.
  Fact 2.1: The Moana soundtrack is from 2016.
  Fact 2.2: Lin-Manuel Miranda is set to star in Disney's upcoming film Mary Poppins Returns.

Retrieved items:
<item 1>
Hamilton (album)
Hamilton is the cast album to the 2015 musical "Hamilton".

 The musical is based on the 2004 biography of Alexander Hamilton written by Ron Chernow, with music, lyrics, and book by Lin-Manuel Miranda.

 The recording stars Lin-Manuel Miranda, Leslie Odom Jr., Phillipa Soo, Renée Elise Goldsberry, Christopher Jackson, Daveed Diggs, Anthony Ramos, Okieriete
Onaodowan, Jasmine Cephas Jones, and Jonathan Groff.

 It also features Jon Rua, Thayne Jasperson, Sydney James Harcourt, Ariana DeBose, and Sasha Hutchings.

 It achieved the largest first week sales for a digital cast album and is the highest-charting cast album since 1963.

 It was the highest-selling Broadway cast album of 2015 and peaked at number one on the Rap Album chart, the first cast album to ever do so.
</item 1>

<item 2>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
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
Cars 2
Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the sequel to 2006's "Cars", and features the voices of Owen Wilson, Larry the Cable Guy, Michael Caine, Emily Mortimer, John Turturro, and Eddie Izzard.

 In the film, race car Lightning McQueen and tow truck Mater head to Japan and Europe to compete in the World Grand Prix, but Mater becomes sidetracked with
international espionage.

 The film was directed by John Lasseter, written by Ben Queen, and produced by Denise Ream.
</item 4>

<item 5>
Cabinet Battle
"Cabinet Battle #1", "Cabinet Battle #2", and "Cabinet Battle #3" are songs written for Act II of the musical "Hamilton", based on the life of Alexander
Hamilton, which premiered on Broadway in 2015.

 Lin-Manuel Miranda wrote both the music and lyrics to the song.
</item 5>

<item 6>
Your Obedient Servant (song)
"Your Obedient Servant" is the twentieth song from Act 2 of the musical "Hamilton", based on the life of Alexander Hamilton, which premiered on Broadway in
2015.

 Lin-Manuel Miranda wrote both the music and lyrics to the song.

 The song is sung by the characters of Hamilton and Aaron Burr, originally performed by Miranda and Leslie Odom Jr., respectively.
</item 6>

<item 7>
Toy Story 3
Toy Story 3 is a 2010 American 3D computer-animated comedy-drama film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the third installment in Pixar's "Toy Story" series, and the sequel to 1999's "Toy Story 2".

 It was directed by Lee Unkrich, the editor of the first two films and the co-director of "Toy Story 2", written by Michael Arndt, while Unkrich wrote the story
along with John Lasseter and Andrew Stanton, respectively director and co-writer of the first two films.

 The plot focuses on the toys Woody, Buzz Lightyear, and their friends dealing with an uncertain future as their owner, Andy, prepares to leave for college.
</item 7>

<item 8>
Cars 2 (video game)
Cars 2 (also known as Cars 2: The Video Game) is a racing game developed by Avalanche Software.

 It is based on the film "Cars 2", the sequel to the 2006 film "Cars".

 Originally announced at E3 2011, the game was released by Disney Interactive on major platforms in North America on June 21, 2011, and in Australia 2 days
later.

 The game was released in Europe on July 22, 2011.

 The PlayStation 3 version of the game was reported to feature stereoscopic 3D gameplay.

 A Nintendo 3DS version was released on November 1, 2011 (the same day the film was released on DVD and Blu-ray).

 A PlayStation Portable version was released in North America on November 8, 2011.

 It was also released in Europe on November 4, 2011.

 It was the last Pixar game to be released for the PlayStation Portable.
</item 8>

<item 9>
Moana (soundtrack)
Moana: Original Motion Picture Soundtrack is the soundtrack to the 2016 Disney animated film "Moana".

 The soundtrack was released by Walt Disney Records on November 19, 2016.

 It features songs written by Lin-Manuel Miranda, Mark Mancina and Opetaia Foa'i, with lyrics in English, Samoan, Tokelauan, and Tuvalu.

 The two-disc deluxe edition includes the score, which was composed by Mancina, as well as demos, outtakes and instrumental karaoke tracks.
</item 9>

<item 10>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 10>
=== END ITEM id=7ca8f0b138dc46b0 ===

=== ITEM id=f05d45f28c4527b0 ===
Gold sentences and facts:
Sentence 0:  Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.
  Fact 0.0: Watkins was born in Des Moines, Iowa.
  Fact 0.1: Watkins rose to fame in the early 1990s.
  Fact 0.2: Watkins rose to fame as a member of the girl-group TLC.
Sentence 1:  She is best known as the former main vocalist and member of South Korean girl group T-ara.
  Fact 1.0: Watkins is best known as the former main vocalist of the South Korean girl group T-ara.
  Fact 1.1: Watkins is best known as the former member of the South Korean girl group T-ara.

Retrieved items:
<item 1>
TLC (group)
TLC is an American girl group whose original line-up consisted of Tionne "T-Boz" Watkins, Lisa "Left Eye" Lopes and Rozonda "Chilli" Thomas.

 Formed in Atlanta, Georgia in 1990, the group was very successful during the 1990s despite numerous spats with the law, each other, and the group's record
label and management.

 They scored nine top-ten hits on the "Billboard" Hot 100, including four number-one singles "Creep", "Waterfalls", "No Scrubs", and "Unpretty".

 The group also recorded four multi-platinum albums, including "CrazySexyCool" (1994) which still remains the only album by a female group to receive a diamond
certification from the Recording Industry Association of America (RIAA).

 TLC also became the first R&B group in history to receive Million certification from the Recording Industry Association of Japan (RIAJ) for "FanMail" (1999).
</item 1>

<item 2>
Tionne Watkins
Tionne Tenese Watkins (born April 26, 1970), better known by her stage name T-Boz, is an American singer, songwriter, actress, author, and executive producer.

 Born in Des Moines, Iowa, Watkins rose to fame in the early 1990s as a member of the girl-group TLC.

 She has won four Grammy Awards for her work with TLC.
</item 2>

<item 3>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 3>

<item 4>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 4>

<item 5>
R U the Girl
R U the Girl (also known as R U the Girl with T-Boz and Chilli) is an American reality series that aired on UPN in 2005.

 The series featured Tionne "T-Boz" Watkins and Rozonda "Chilli" Thomas, the remaining members of the all-girl R&B group TLC whose former member, Lisa "Left
Eye" Lopes, died in a car crash in Honduras on April 25, 2002.

 Initially promoted as a contest to permanently replace Lopes 3 years after her death by TLC themselves, both Watkins and Thomas admitted that the winner of the
contest would not be joining TLC full-time and would not be a full-time replacement member; the winner would only provide guest vocals on a new single by the
duo.
</item 5>

<item 6>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 6>

<item 7>
Kim Tae-yeon (actress)
Kim Tae-yeon (born January 3, 1976) is a South Korean actress.

 She began her entertainment career as a model, winning Model Line's 40th Fashion Model contest in 1996 and the Pantene Model contest sponsored by Ford Models
in 2000.

 Kim made her film debut in the highly controversial film "Lies" in 1999.
</item 7>

<item 8>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 8>

<item 9>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 9>

<item 10>
Rain (Taeyeon song)
"Rain" is a song by South Korean singer Kim Tae-yeon, a member of the South Korean girl group Girls' Generation.

 It was released digitally by S.M. Entertainment on February 3, 2016.
</item 10>
=== END ITEM id=f05d45f28c4527b0 ===

=== ITEM id=4436a604c598879c ===
Gold sentences and facts:
Sentence 0: "For the sedan variant of the car, see Volvo S40."
  Fact 0.0: The sedan variant of the car is described under Volvo S40.
Sentence 1: The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
  Fact 1.0: The Volvo S40 is a compact luxury car.
  Fact 1.1: The Volvo S40 is marketed by Volvo Cars.
  Fact 1.2: The Volvo S40 is produced by Volvo Cars.
  Fact 1.3: Volvo Cars is a Swedish manufacturer.
  Fact 1.4: The Volvo S40 was marketed and produced from 1995 to 2012.
  Fact 1.5: The Volvo S40 spanned two generations.

Retrieved items:
<item 1>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 1>

<item 2>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 2>

<item 3>
Volvo V50
"For the sedan variant of the car, see Volvo S40."
</item 3>

<item 4>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 4>

<item 5>
Volvo 440/460
The Volvo 440 and 460 are versions of a small family car produced by the Swedish manufacturer Volvo between 1987 and 1996.

 The 440 was introduced in 1987, whilst the 460 followed in 1989.

 They were built at the NedCar factory in Born, The Netherlands.

 The 440 was a five door front wheel drive hatchback and the 460 a saloon.

 They shared many components with the already successful Volvo 480, including floorpan, front and rear suspension, engines, transmission and braking systems.
</item 5>

<item 6>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 6>

<item 7>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 7>

<item 8>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 8>

<item 9>
Volvo 480
The Volvo 480 is a compact car that was produced in Born, Netherlands, by Volvo from 1986 to 1995.

 It was the first front-wheel drive car made by the automaker.

 The 480 was available in only one body style on a automobile platform related to the Volvo 440/460 five door hatchback and four door sedan models.
</item 9>

<item 10>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 10>
=== END ITEM id=4436a604c598879c ===

=== ITEM id=f515795edb56bcb6 ===
Gold sentences and facts:
Sentence 0: "Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".
  Fact 0.0: 'Empty Orchestra' is the fourth episode of the third series of Inside No. 9.
  Fact 0.1: Inside No. 9 is a British dark comedy anthology television programme.
Sentence 1:  Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.
  Fact 1.0: 'Empty Orchestra' was written by Steve Pemberton.
  Fact 1.1: 'Empty Orchestra' was written by Reece Shearsmith.
  Fact 1.2: 'Empty Orchestra' was directed by Guillem Morales.
  Fact 1.3: 'Empty Orchestra' first aired on 7 March 2017.
  Fact 1.4: 'Empty Orchestra' first aired on BBC 2.
Sentence 2:  The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).
  Fact 2.0: The episode 'Empty Orchestra' is set in a karaoke booth.
  Fact 2.1: The episode 'Empty Orchestra' follows a group of work colleagues.
  Fact 2.2: The work colleague Greg is shown as Shearsmith.
  Fact 2.3: The work colleague Fran is shown as Sarah Hadland.
  Fact 2.4: The work colleague Connie is shown as Tamzin Outhwaite.
  Fact 2.5: The work colleague Janet is shown as Emily Howlett.
  Fact 2.6: The work colleague Duane is shown as Javone Prince.
  Fact 2.7: The colleagues are celebrating the promotion of Roger.
  Fact 2.8: Roger is shown as Pemberton.
Sentence 3:  Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and television productions, including army series "Red Cap" and crime drama "New Tricks".
  Fact 3.0: The performer described in this sentence came to national notice playing Melanie Owen in the BBC One soap opera EastEnders from 1998 until 2002.
  Fact 3.1: The performer has since starred in theatre productions.
  Fact 3.2: The performer has since starred in television productions.
  Fact 3.3: The performer has starred in the army series Red Cap.
  Fact 3.4: The performer has starred in the crime drama New Tricks.

Retrieved items:
<item 1>
Vital Signs (TV series)
Vital Signs is a British television drama series airing on ITV from 2006.

 It stars Tamzin Outhwaite as a supermarket check-out operator who decides to become a doctor.

 The series co-stars William Beck, Fraser Ayres, Eve Best, Claudie Blakley, Lucinda Dryzek, Beth Goddard, Alfie Hunter, Brooke Kinsella, Harry Lloyd, Peter Rnic
and Steven Waddington.

 The filming of the show is based in numerous London hospitals and medical schools; predominantly the show has been shot in St George's Hospital and Medical
School.
</item 1>

<item 2>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 2>

<item 3>
Paradox (TV series)
Paradox is a 2009 British science fiction police drama, starring Tamzin Outhwaite as Detective Inspector Rebecca Flint.

 Written by Lizzie Mickery and produced by Clerkenwell Films for the BBC, it was filmed and set in Manchester, England.
</item 3>

<item 4>
Out of Control (2002 film)
Out of Control is a 2002 British television film written and directed for BBC by Dominic Savage.

 It stars Tamzin Outhwaite as Shelley Richards, an impoverished single mother whose son, Dean (Danny Young), is involved in a string of crimes that lead to his
incarceration in a young offenders institute (YOI).

 David Morrissey plays prison warden Mike, who tries to keep Dean out of trouble but has difficultly watching him all the time.

 After continuous bullying at the YOI, Dean is no longer able to cope and makes a fatal decision.

 It is the third in a loose trilogy of films by Savage about social deprivation, following "Nice Girl" (2000) and "When I Was Twelve" (2001).
</item 4>

<item 5>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 5>

<item 6>
Radio Cape Cod
Radio Cape Cod is a 2008 film directed and produced by Andrew Silver.

 It stars Tamzin Outhwaite as a radio interviewer coming to terms with the loss of her husband and a new love entering her life.

 She also must cope with her teenage daughter, played by Tamzin Merchant, who is experiencing her first love.

 The film also stars O. T. Fagbenle.

 The film won several festival awards.
</item 6>

<item 7>
Tamzin Outhwaite
Tamzin Maria Outhwaite ( ; born 5 November 1970) is a British actress from London.

 Since coming to national notice playing Melanie Owen in the BBC One soap opera "EastEnders" from 1998 until 2002, she has since starred in both theatre and
television productions, including army series "Red Cap" and crime drama "New Tricks".
</item 7>

<item 8>
John Altman (actor)
John Clarkson Stewart (born 2 March 1952), known as John Altman, is an English actor and singer, perhaps best known for playing "Nasty" Nick Cotton in the
popular BBC soap opera "EastEnders".

 He was among the show's original cast members appearing in the very first episode in February 1985 and appeared on the show on and off as a recurring
character.

 His character was killed off in the 30th anniversary episode of the show which aired in February 2015.

 Altman has also appeared in several other television series and appeared in many stage productions.

 In 2010 he became the new frontman of the band Heavy Metal Kids following the death of former frontman Gary Holton in 1985.
</item 8>

<item 9>
Melanie Owen
Melanie Jane "Mel" Owen (also Healy and Beale) is a fictional character from the BBC soap opera "EastEnders", played by Tamzin Outhwaite.

 She appeared from 19 October 1998 to 12 April 2002 when Outhwaite opted to leave the role.

 Mel was introduced by executive producer Matthew Robinson and became a prominent character during her tenure in the soap.
</item 9>

<item 10>
Empty Orchestra
"Empty Orchestra" is the fourth episode of the third series of the British dark comedy anthology television programme "Inside No. 9".

 Written by Steve Pemberton and Reece Shearsmith and directed by Guillem Morales, the episode first aired on 7 March 2017, on BBC 2.

 The episode is set in a karaoke booth, and follows a group of work colleagues—Greg (Shearsmith), Fran (Sarah Hadland), Connie (Tamzin Outhwaite), Janet (Emily
Howlett) and Duane (Javone Prince)—celebrating the promotion of Roger (Pemberton).

 Rebekah Hinds also stars.
</item 10>
=== END ITEM id=f515795edb56bcb6 ===

=== ITEM id=c4483ace7cc16eeb ===
Gold sentences and facts:
Sentence 0:  He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
  Fact 0.0: The person in this sentence succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
Sentence 1: Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death in 1250.
  Fact 1.0: Eric IV is also known as Eric Ploughpenny.
  Fact 1.1: Eric IV is also known as Eric Plowpenny.
  Fact 1.2: Eric IV's Danish name is Erik Plovpenning.
  Fact 1.3: Eric IV's dates are 1216 to 10 August 1250.
  Fact 1.4: Eric IV was king of Denmark from 1241.
  Fact 1.5: Eric IV was king of Denmark until his death in 1250.

Retrieved items:
<item 1>
King Kang of Chu
King Kang of Chu (, died 545 BC) was from 559 to 545 BC the king of Chu, a major power during the Spring and Autumn period of ancient China.

 Born Xiong Zhao (), he succeeded his father King Gong of Chu.

 His reign was marked by constant wars with Jin, Chu's traditional enemy, and Wu, its new enemy.

 He ruled for 15 years and was succeeded by his son, Jia'ao, who would four years later be murdered by King Kang's younger brother King Ling of Chu.
</item 1>

<item 2>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 2>

<item 3>
Eric IV of Denmark
Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death
in 1250.

 He was the son of King Valdemar II by his wife, Berengaria of Portugal, and brother of King Abel and King Christopher I.
</item 3>

<item 4>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 4>

<item 5>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 5>

<item 6>
Harald II ruled Denmark (1014–18), succeeding father Sweyn Forkbeard while brother Cnut conquered England; Cnut succeeded him. Christopher I (1252–59), son of
Valdemar II, succeeded Abel. Epirus’s Ptolemy briefly ruled after Pyrrhus II, dying on campaign (or assassinated); niece Deidamia followed. Chu’s King Kang
(559–545 BC) fought Jin and Wu; son Jia’ao succeeded
</item 6>

<item 7>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 7>

<item 8>
Christopher I of Denmark
Christopher I (Danish: "Christoffer I" ) (1219 – 29 May 1259) was King of Denmark between 1252 and 1259.

 He was the son of Valdemar II of Denmark by his wife, Infanta Berengária of Portugal.

 He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.

 Christopher was elected King upon the death of his older brother Abel in the summer of 1252.

 He was crowned at Lund Cathedral on Christmas Day 1252.
</item 8>

<item 9>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 9>

<item 10>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 10>
=== END ITEM id=c4483ace7cc16eeb ===

=== ITEM id=169ecc2df2533458 ===
Gold sentences and facts:
Sentence 0: Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that confirmed Trevi's status as a gay icon.
  Fact 0.0: Todos Me Miran ('Everyone looks at me') is a single from the Mexican artist Gloria Trevi.
  Fact 0.1: Todos Me Miran reached number 32 on Latin charts.
  Fact 0.2: Todos Me Miran became a club anthem.
  Fact 0.3: Todos Me Miran confirmed Trevi's status as a gay icon.
Sentence 1: Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
  Fact 1.0: Gloria Trevi was born Gloria de los Angeles Trevino Ruiz.
  Fact 1.1: Gloria Trevi was born on February 15, 1968.
  Fact 1.2: Gloria Trevi is a Mexican singer.
  Fact 1.3: Gloria Trevi is a Mexican songwriter.

Retrieved items:
<item 1>
Gloria Trevi
Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
</item 1>

<item 2>
A Cambio de Que
"A Cambio de Que" (In Exchange for What) is a song written by Xavier Santos Córtes and first performed by Mexican singer Marisela from her studio album
"Completamente Tuya".

 The song has been covered by several artists which have ranked on the music charts.

 Latin pop band Corvo covered the song in 1999 which reached #18 on the Hot Latin Songs chart in the United States.

 Mexican band Palomo's cover peaked at #31 on the Regional Mexican Airplay chart.

 American singer Huey Dunbar recorded "A Cambio de Que" on his debut studio album "Yo Si Me Enamore" (2001).

 This version peaked at #16 on the Tropical Songs chart.

 Mexican duranguense band Alacranes Musical covered the song on the album of the same name in 2004 which reached #25 on the Regional Mexican Airplay chart.
</item 2>

<item 3>
Tu Ángel de la Guarda
Tu Ángel de la Guarda ("Your guardian angel") is Gloria Trevi's second album, and it contained one of her signature songs and her most widely known hit, "Pelo
Suelto".

 It also contained other hit songs such as "Tu angel de la guarda", "Ya no", "Virgen de las virgenes", and "Hoy me ire de casa".

 This album was very controversial different from other artists' albums such as Lucero.

 "Virgen de las virgenes" mocked girls who said they were virgins but had actually lost their virginity, and "¡Ya no!"

 went against the machismo movement of México.

 Following the release of the album, Gloria was working hard in promoting radio, television and print media.

 Trevi first traveled abroad visiting American Union countries, Puerto Rico, Argentina, Colombia, Guatemala, Venezuela, Costa Rica, and received a Gold and
Platinum for high sales achieved.
</item 3>

<item 4>
La Trayectoria (Gloria Trevi album)
La Trayectoria (In English: The Trajectory) is a 2006 double live CD and multimedia DVD release by the Mexican singer Gloria Trevi.

 The DVD includes scenes from parts of the tour from the Palacio de los Deportes of the Mexico City and The Arena Monterrey and the CD includes four new songs
like: "Todos Me Miran" (They All Look At Me) is the first single compilation study of Mexican singer "Gloria Trevi", this is an unreleased song from the album.
</item 4>

<item 5>
Todos Los Romances
Todos Los Romances (English: "All the Romances" ) is a three-disc compilation album by Mexican singer Luis Miguel.

 Released on 11 August 1998 by WEA Latina, the record features the three previously released "Romance"-themed albums in which Miguel covered classic boleros in
each of them: "Romance" (1991), "Segundo Romance" (1994), and "Romances" (1997).

 An editor for AllMusic rated the album four of five stars.

 Commercially, "Todos Los Romances" peaked at number four in Spain and was certified double Platinum in the country.

 It also achieved Gold status in Argentina and peaked at number 12 on the "Billboard"'s Top Latin Albums in the United States.
</item 5>

<item 6>
Premio Lo Nuestro 2003
In the Regional Mexican, Pilar Montenegro took three awards for Regional Mexican Song of the Year, Pop Song of the Year ("Quitame Ese Hombre"), and for Best
Regional Mexican Female Artist.

 In the tropical genre, Celia Cruz took home with four great awards of the night for Best Salsa Performance, Best Tropical Female Artist, Best Tropical Song of
the year and Tropical Album of the Year.

 At the night, the greatest performance was a medley of top Latin hits from the last 15 years, performed by the artists that made them famous, including Vikki
Carr, Son by Four, Los Ilegales, La Mafia, Luis Enrique, Wilfrido Vargas and Olga Tañón.

 There was a great tribute to Celia Cruz by the world-famous salsa group "Fania All-Stars", of which Cruz was a member during the 1970s, reunited for an
exclusive performance that rocked the house.
</item 6>

<item 7>
No eres tú, soy yo
No eres tú, soy yo (English: It's not you, it's me ) is a 2010 Mexican romantic comedy directed by Alejandro Springall starring Eugenio Derbez, Alejandra Barros
and Martina García and based on the Argentinean film "No sos vos, soy yo" (2004), written by Juan Taratuto.

 It was produced by Matthias Eherenberg and filmed in Mexico City.
</item 7>

<item 8>
De Contrabando
"De Contrabando" ("Smuggled") is a song written by Joan Sebastian.

 It was first performed by Mexican singer Imelda, and included on her album "Aparentemente", released in 1991.

 Six years later, was performed by fellow Mexican singer Zayda Y Los Culpables and released by Musart Records as the second single from her self-titled album in
1997.

 Mexican-American singer Jenni Rivera recorded a cover version for her studio album "Parrandera, Rebelde y Atrevida".

 "De Contrabando" was released as the second single from the album in 2006.

 In the United States, the song peaked at number fourteen on the "Billboard" Hot Latin Songs chart and number one on the "Billboard" Regional Mexican Songs
chart.

 The song was nominated for a Lo Nuestro Award for Regional Mexican Song of the Year at the 2007 Lo Nuestro Awards.

 Rivera was also recognized for Regional Mexican Female Artist of the Year.

 Rivera was also awarded Regional Mexican Airplay Song of the Year for at the 2007 Latin "Billboard" Music Awards.
</item 8>

<item 9>
Mexico Espanol Airplay
Mexico Espanol Airplay is a record chart published weekly by "Billboard" magazine for Spanish language singles released in Mexico.

 According to "Billboard"' s electronic database, the first chart was published on October 1, 2011, with the track "Amor Clandestino" by Mexican band Maná at
number one; in the same year, fellow Mexican duo Jesse & Joy reached the top of the chart with their single "¡Corre!

", the first of their six number-one singles in the chart, the most for any act. "

¡Corre!"

 also won the Latin Grammy Awards for Record of the Year and Song of the Year in 2012.

 Starting on December 1, 2012, Mexican singer Thalía spent 10 non-consecutive weeks at number-one with "Manías", the first single from the album "Habítame
Siempre", which won the Lo Nuestro Award for Pop Album of the Year. "

Hoy Tengo Ganas de Ti" by Mexican artist Alejandro Fernández and American singer Christina Aguilera reached number-one in 2013, and also peaked at number four
in Spain.
</item 9>

<item 10>
Todos Me Miran
Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that
confirmed Trevi's status as a gay icon.

 The song, as interpreted in the music video, is about a young man who dares to crossdress in spite of society's opinions.
</item 10>
=== END ITEM id=169ecc2df2533458 ===

=== ITEM id=06549fbf4735adb4 ===
Gold sentences and facts:
Sentence 0: David Michael Draiman (born March 13, 1973) is an American songwriter and the vocalist for the band Disturbed as well as for the band Device.
  Fact 0.0: David Michael Draiman was born on March 13, 1973.
  Fact 0.1: David Michael Draiman is an American songwriter.
  Fact 0.2: David Michael Draiman is the vocalist for the band Disturbed.
  Fact 0.3: David Michael Draiman is the vocalist for the band Device.
Sentence 1: Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and founding members of the rock band Chicago.
  Fact 1.0: Terry Alan Kath lived January 31, 1946 to January 23, 1978.
  Fact 1.1: Terry Alan Kath was an American musician.
  Fact 1.2: Terry Alan Kath was an American songwriter.
  Fact 1.3: Terry Alan Kath was best known as the original guitarist of the rock band Chicago.
  Fact 1.4: Terry Alan Kath was one of the lead singers of the rock band Chicago.
  Fact 1.5: Terry Alan Kath was a founding member of the rock band Chicago.

Retrieved items:
<item 1>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 1>

<item 2>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 2>

<item 3>
Katherine Dienes
Katherine Dienes (born 1970) is a New Zealand-born organist, conductor and composer.

 She is currently Organist and Master of the Choristers at Guildford Cathedral and is the first woman to hold the most senior musical post in a Church of
England cathedral.

 Her husband is Patrick Williams - librarian of the Royal Phillharmonic Orchestra, and they have a daughter, Hannah, who is currently a chorister at Guildford
Cathedral
</item 3>

<item 4>
David Draiman
David Michael Draiman (born March 13, 1973) is an American songwriter and the vocalist for the band Disturbed as well as for the band Device.

 Draiman is known for his distorted voice and percussive singing style.

 In November 2006, Draiman was voted number 42 on the "Hit Parader"’s ""Top 100 Metal Vocalists of All Time"".

 Draiman has written some of Disturbed's most successful singles, such as "Stupify", "Down with the Sickness", "Indestructible", and "Inside the Fire".
</item 4>

<item 5>
Device (metal band)
Device was an industrial metal band started by David Draiman, frontman of the heavy metal group Disturbed.

 Draiman was approached by Geno Lenardo, former guitarist of Filter, and together they started to work on new material.

 The two started work on a debut album in June 2012.

 The result of those sessions, "Device", was released on April 9, 2013.

 The first single and third track, "Vilify" was released to radio ahead of the album on February 19, 2013, alongside its first music video, directed by P. R.
Brown.

 The second single and the introductory track, "You Think You Know", was released on June 11, 2013, alongside its music video, directed once again by Brown.
</item 5>

<item 6>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 6>

<item 7>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 7>

<item 8>
Pantera
Pantera was an American heavy metal band from Arlington, Texas.

 The group was formed in 1981 by the Abbott brothers – drummer Vinnie Paul and guitarist Dimebag Darrell – along with lead vocalist Terry Glaze.

 Bassist Rex Brown joined the band the following year, replacing Tommy D. Bradford, who was the unofficial original.

 Having started as a glam metal band, Pantera released four albums during the 1980s.

 Looking for a new and heavier sound, Pantera replaced Glaze with Phil Anselmo in late 1986 and released "Power Metal" in 1988.

 With its fifth album, 1990's "Cowboys from Hell", Pantera introduced a groove metal sound.

 Pantera's sixth album, 1992's "Vulgar Display of Power", exhibited an even heavier sound. "

Far Beyond Driven" (1994) debuted at number one on the "Billboard" 200.
</item 8>

<item 9>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 9>

<item 10>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 10>
=== END ITEM id=06549fbf4735adb4 ===
