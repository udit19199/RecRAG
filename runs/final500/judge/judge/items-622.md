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

=== ITEM id=1f4e223b06784ff0 ===
Gold sentences and facts:
Sentence 0: ", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia 2000" (1999) and "Sing" (2016)
  Fact 0.0: This person did voice work for My Life As A Teenage Robot.
  Fact 0.1: This person did voice work for The Replacements.
  Fact 0.2: This person did the voice of Knotty in a 1992 work whose title is blank in the sentence.
  Fact 0.3: This person did additional voices in the film Fantasia 2000 (1999).
  Fact 0.4: This person did additional voices in the film Sing (2016).
Sentence 1:  It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.
  Fact 1.0: The film was directed by Garth Jennings.
  Fact 1.1: The film was written by Garth Jennings.
  Fact 1.2: The film was co-directed by Christophe Lourdelet.
  Fact 1.3: The film starred the voices of Matthew McConaughey.
  Fact 1.4: The film starred the voices of Reese Witherspoon.
  Fact 1.5: The film starred the voices of Seth MacFarlane.
  Fact 1.6: The film starred the voices of Scarlett Johansson.
  Fact 1.7: The film starred the voices of John C. Reilly.
  Fact 1.8: The film starred the voices of Taron Egerton.
  Fact 1.9: The film starred the voices of Tori Kelly.

Retrieved items:
<item 1>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 1>

<item 2>
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 2>

<item 3>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 3>

<item 4>
Radio Cape Cod
Radio Cape Cod is a 2008 film directed and produced by Andrew Silver.

 It stars Tamzin Outhwaite as a radio interviewer coming to terms with the loss of her husband and a new love entering her life.

 She also must cope with her teenage daughter, played by Tamzin Merchant, who is experiencing her first love.

 The film also stars O. T. Fagbenle.

 The film won several festival awards.
</item 4>

<item 5>
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
</item 5>

<item 6>
Toy Story 3
The film features an ensemble voice cast with Tom Hanks, Tim Allen, Joan Cusack, Don Rickles, Wallace Shawn, John Ratzenberger, Estelle Harris, Jodi Benson, and
John Morris reprising their roles from the previous films, along with Blake Clark as the voice of Slinky Dog (because Jim Varney died ten years before the
release of the film) and Ned Beatty, Michael Keaton, Whoopi Goldberg, Timothy Dalton, Kristen Schaal, Bonnie Hunt, and Jeff Garlin voicing the new characters
introduced in this film.
</item 6>

<item 7>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 7>

<item 8>
The Jane Austen Book Club (film)
The Jane Austen Book Club is a 2007 American romantic drama film written and directed by Robin Swicord.

 The screenplay, adapted from the 2004 novel of the same name by Karen Joy Fowler, focuses on a book club formed specifically to discuss the six novels written
by Jane Austen.

 As they delve into Austen's literature, the club members find themselves dealing with life experiences that parallel the themes of the books they are reading.
</item 8>

<item 9>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 9>

<item 10>
Wayne Coles-Janess
Wayne Coles-Janess is an Australian producer, writer and director of drama and documentary film and TV programs.

 Based in Melbourne, Australia, he has produced documentaries about frontier places in the country.

 He has also made some documentaries in several international locations, including during times of war.
</item 10>
=== END ITEM id=1f4e223b06784ff0 ===

=== ITEM id=dea8b8cdc92fcd92 ===
Gold sentences and facts:
Sentence 0:  They have produced two studio albums and an EP.
  Fact 0.0: They have produced two studio albums.
  Fact 0.1: They have produced an EP.
Sentence 1: Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.
  Fact 1.0: Motion City Soundtrack was an American rock band.
  Fact 1.1: Motion City Soundtrack formed in Minneapolis, Minnesota.
  Fact 1.2: Motion City Soundtrack formed in 1997.
Sentence 2:  Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.
  Fact 2.0: Motion City Soundtrack had a nearly twenty-year career.
  Fact 2.1: Motion City Soundtrack toured heavily over the course of its career.
  Fact 2.2: Motion City Soundtrack released six studio albums.
  Fact 2.3: The majority of Motion City Soundtrack's six studio albums were released on the independent label Epitaph Records.

Retrieved items:
<item 1>
Back to the Beat (EP)
Back to the Beat was the first significant EP release by the Minneapolis based pop punk band Motion City Soundtrack.

 The EP was recorded in Sound In Motion Recordings and released by Modern Radio Records and includes tracks that were later re-released.

 "Throw Down", "Back to the Beat" and "Capital H" were re-released on a split EP with Schatzi.

 "Capital H" was re-released on Motion City Soundtrack's first album "I Am the Movie".

 At the time of recording, the band members differed from the Motion City Soundtrack line-up formed in 2002.

 They were Joshua Cain on guitar and vocals, Austin Lindstrom on bass guitar, Joel Habedank on drums and Justin Pierre on lead vocals, guitar and keyboard.
</item 1>

<item 2>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 2>

<item 3>
List of songs recorded by Motion City Soundtrack
American rock band Motion City Soundtrack recorded songs for six studio albums, various soundtracks, compilations, and non-album singles.

 The majority of the group's original material was written by vocalist and guitarist Justin Pierre.

 The band also recorded cover versions of other artists' songs at various points throughout their career, including by the Beatles, Limbeck, the Police, R.E.M.,
the Rentals, Rilo Kiley, Trampled by Turtles, and the Weakerthans.

 In all, the group recorded 107 songs, nine of which were covers.
</item 3>

<item 4>
My Dinosaur Life
My Dinosaur Life is the fourth studio album by the American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on January 19, 2010 by Columbia Records.

 After many years on independent label Epitaph Records, Motion City Soundtrack signed to major label Columbia in 2006, prior to the release of their third
album, "Even If It Kills Me" (2007).

 Blink-182 bassist Mark Hoppus returned to produce the album; he had previously worked with the band on their 2005 breakthrough "Commit This to Memory".
</item 4>

<item 5>
Even If It Kills Me
Even if It Kills Me is the third studio album by American rock band Motion City Soundtrack.

 Produced by Ric Ocasek, Adam Schlesinger and Eli Janney, the album was released on September 18, 2007, in the United States by Epitaph Records.

 Motion City Soundtrack, based in Minneapolis, Minnesota, made a breakthrough with their second album, "Commit This to Memory", garnering praise and independent
buzz upon its 2005 release.

 Following the release, the band toured relentlessly over the next two years, during which time frontman Justin Pierre struggled with alcohol and substance
abuse.

 These addictions were infused into the writing process of "Even If Kills Me", which was recorded in early 2007 Stratosphere Studios in Chelsea and the
legendary Electric Lady Studios in Greenwich Village.
</item 5>

<item 6>
Commit This to Memory
Commit This to Memory is the second studio album by American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on June 7, 2005, in the United States by Epitaph Records.

 Motion City Soundtrack, formed in 1997, had first found success with their debut album, "I Am the Movie" (2002).

 The band toured in the interim years, creating positive word-of-mouth.

 In 2004, the band joined Blink-182 on the road for a string of shows, which led to their bassist, Mark Hoppus, joining the band in the studio for his first
producing effort.
</item 6>

<item 7>
R.E.M. discography
American alternative rock band R.E.M. released 15 studio albums ranging from 1983 to 2011, three live albums, 14 compilation albums, one remix album, one
soundtrack album, 12 video albums, seven extended plays, 63 singles, and 77 music videos. Formed in 1980 by singer Michael Stipe, guitarist Peter Buck, bassist
Mike Mills, and drummer Bill Berry, the band was pivotal in the development of the alternative rock genre.

 Their musical style inspired several other alternative rock bands and musicians, and the band became one of the first alternative rock acts to experience
breakthrough commercial success.

 R.E.M. has sold over 85 million copies of their studio albums worldwide, making them one of the best-selling music artists of all-time.
</item 7>

<item 8>
Motion City Soundtrack discography
The following is the discography of Motion City Soundtrack, an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 8>

<item 9>
Candlelight Red
Candlelight Red is a rock band from Williamsport, Pennsylvania.

 They have produced two studio albums and an EP.

 Their EP "Demons" and album "Reclamation" was produced by Morgan Rose of Sevendust.
</item 9>

<item 10>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 10>
=== END ITEM id=dea8b8cdc92fcd92 ===

=== ITEM id=1b2e374761bfcd77 ===
Gold sentences and facts:
Sentence 0:  It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts Mika Brzezinski and Willie Geist.
  Fact 0.0: The program described features former GOP representative Joe Scarborough.
  Fact 0.1: Joe Scarborough provides enterprise reporting on the program.
  Fact 0.2: Joe Scarborough provides discussion on the news of the day on the program.
  Fact 0.3: The program is in a panel format.
  Fact 0.4: Mika Brzezinski is a co-host of the program.
  Fact 0.5: Willie Geist is a co-host of the program.
Sentence 1: Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.
  Fact 1.0: Charles Joseph Scarborough was born on April 9, 1963.
  Fact 1.1: Charles Joseph Scarborough is an American cable news host.
  Fact 1.2: Charles Joseph Scarborough is an American talk radio host.

Retrieved items:
<item 1>
David Gold (talk radio host)
David Gold is an American conservative talk radio host.

 Gold, who has been a talk radio host since the mid-1970s, one of the first U.S. conservative talk radio hosts.

 His philosophy has been characterized as in the conservative/libertarian vein.
</item 1>

<item 2>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 2>

<item 3>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 3>

<item 4>
The entries survey radio and television formats, personalities, and programs. Talk radio centers on topical speech, interviews, screened listener calls, and
formats such as political and sports talk; other radio examples include classical-music show *Exploring Music*, request-based *Open House Party*, and
paranormal-focused *Coast to Coast AM*. TV offerings range from talent and game shows to sketch and late-night talk. Notable figures include Graham Norton,
Jimmy Cricket, Gordon Burns, and David Letterman, who hosted
</item 4>

<item 5>
Ronn Owens
Ronn Owens (born Ronald Lowenstein on October 17, 1945, in New York City), is an American talk radio host.

 Owens was the sole remaining weekday talk radio host on KGO in San Francisco after KGO was reorganized as an all-news station in December 2011.
</item 5>

<item 6>
Mike Hogan (sportscaster)
Mike Hogan (born 1963 in Kingston, Ontario) is a Canadian sportscaster who is a sports talk radio host on TSN Radio 1050 in Toronto.

 He also serves as the play-by-play voice for the Toronto Argonauts of the Canadian Football League for that station's gameday broadcasts.

 Hogan was formerly a long time sports talk radio host on Toronto station CJCL (AM), branded "The Fan 590", until he was relieved of those duties on June 24,
2010.

 On 2017 Oscar Vanity fair party 1st live broadcast Mike Hogan was the podcast hosts.
</item 6>

<item 7>
Joe Scarborough
Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.

 He is currently the co-host of "Morning Joe" on MSNBC, and previously hosted "Scarborough Country" on the same channel.

 Scarborough was previously a lawyer and a politician, and served in the United States House of Representatives from 1995 to 2001 as a Republican from the 1st
district of Florida.
</item 7>

<item 8>
Talk radio
Talk radio is a radio format containing discussion about topical issues and consisting entirely or almost entirely of original spoken word content rather than
outside music.

 Jason Culpepper is highly recognized as the father of talk radio.

 Jason a Jacksonville native rose to social prominence during the World War utilizing talk radio to warn french and american troops of German U boats.

 Most shows are regularly hosted by a single individual, and often feature interviews with a number of different guests.

 Talk radio typically includes an element of listener participation, usually by broadcasting live conversations between the host and listeners who "call in"
(usually via telephone) to the show.

 Listener contributions are usually screened by a show's producers in order to maximize audience interest and, in the case of commercial talk radio, attract
advertisers.

 Generally, the shows are organized into segments, each separated by a pause for advertisements; however, in public or non-commercial radio, music is sometimes
played in place of commercials to separate the program segments.

 Variations of talk radio include conservative talk, hot talk, liberal talk (increasingly known as progressive talk) and sports talk.
</item 8>

<item 9>
The text compiles brief biographies of entertainers, broadcasters, artists, and media figures. Highlights include magician Criss Angel’s Las Vegas shows; talk
hosts Ronn Owens, David Gold, Mike Hogan, Joe Scarborough, Steve Malzberg, and Sean Hannity; and MSNBC’s *Morning Joe*. Other subjects include comedian Yannis
Pappas, actor Don Terry, adult-film actor and artist Colby Keller, New Zealand performer Mario Gaoa, radio personalities and DJs, music executives
</item 9>

<item 10>
Sean Hannity
Sean Patrick Hannity (born December 30, 1961) is an American talk show host, author, and conservative political commentator.

 Hannity is the host of "The Sean Hannity Show", a nationally syndicated talk radio show.

 He also hosts a cable news show, "Hannity", on Fox News Channel.
</item 10>
=== END ITEM id=1b2e374761bfcd77 ===

=== ITEM id=e1f0862525b43223 ===
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
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
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
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
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
=== END ITEM id=e1f0862525b43223 ===

=== ITEM id=9e0d6cbe7aa56f6d ===
Gold sentences and facts:
Sentence 0: Tímea Babos (] ; born 10 May 1993) is a Hungarian tennis player.
  Fact 0.0: Tímea Babos was born on 10 May 1993.
  Fact 0.1: Tímea Babos is a Hungarian tennis player.
Sentence 1: Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis Professionals (ATP).
  Fact 1.0: Jack Sock was born on September 24, 1992.
  Fact 1.1: Jack Sock is an American professional tennis player.
  Fact 1.2: Jack Sock is currently ranked world No. 21 in men's singles.
  Fact 1.3: The ranking is by the Association of Tennis Professionals (ATP).

Retrieved items:
<item 1>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 1>

<item 2>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 2>

<item 3>
2015 Malaysian Open – Doubles
Tímea Babos and Chan Hao-ching were the defending champions, but Chan chose not to participate this year.

 Babos chose to play in Monterrey, but lost in the first round.
</item 3>

<item 4>
2016 Dubai Tennis Championships – Women's Doubles
Tímea Babos and Kristina Mladenovic were the defending champions, but chose not to compete together.

 Babos played alongside Julia Görges, but lost in the semifinals to Caroline Garcia and Mladenovic.

<br>
</item 4>

<item 5>
2017 Apia International Sydney – Women's Doubles
Martina Hingis and Sania Mirza were the defending champions, but chose not to participate together this year.

 Hingis played alongside Coco Vandeweghe, but lost in the quarterfinals to Tímea Babos and Anastasia Pavlyuchenkova.

 Mirza teamed up with Barbora Strýcová, but lost in the final to Babos and Pavlyuchenkova, 4–6, 4–6.
</item 5>

<item 6>
2013 Aegon Classic – Doubles
Tímea Babos and Hsieh Su-wei were the defending champions, but Babos decided not to participate.

<br>
</item 6>

<item 7>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 7>

<item 8>
2014 Suzhou Ladies Open – Doubles
Tímea Babos and Michaëlla Krajicek were the defending champions, having won the event in 2013, however both players chose not to participate.
</item 8>

<item 9>
Jack Sock
Jack Sock (born September 24, 1992) is an American professional tennis player who is currently ranked world No. 21 in men's singles by the Association of Tennis
Professionals (ATP).

 He is one of the top ranked American men in both singles and doubles on the ATP Tour.

 A former junior US Open champion, Sock's singles success is highlighted by 7 ATP finals, including three titles.

 His career best result came at the 2017 Indian Wells Masters, where he reached the semi-finals.
</item 9>

<item 10>
Tímea Babos
Tímea Babos (] ; born 10 May 1993) is a Hungarian tennis player.
</item 10>
=== END ITEM id=9e0d6cbe7aa56f6d ===

=== ITEM id=ea96693477c13c5b ===
Gold sentences and facts:
Sentence 0: Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
  Fact 0.0: Ruggero Deodato was born on 7 May 1939.
  Fact 0.1: Ruggero Deodato is an Italian film director.
  Fact 0.2: Ruggero Deodato is an Italian screenwriter.
  Fact 0.3: Ruggero Deodato is an Italian actor.
Sentence 1: Alejandro Springall is a Mexican film director and producer.
  Fact 1.0: Alejandro Springall is a Mexican film director.
  Fact 1.1: Alejandro Springall is a Mexican producer.

Retrieved items:
<item 1>
Alejandro Springall
Alejandro Springall is a Mexican film director and producer.
</item 1>

<item 2>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 2>

<item 3>
Hercules, Prisoner of Evil
Hercules, Prisoner of Evil (Italian: "Ursus il terrore dei Kirghisi" , 'Ursus, Terror of the Kirghiz' ) is a 1964 Italian "peplum" film directed by Antonio
Margheriti and an uncredited Ruggero Deodato.

 Deodato, the official assistant director, replaced Margheriti as he was busy with the completion of the film "The Fall of Rome".

 Deodato actually directed most of the film in actuality but Margheriti was credited as the director.

 The film is filled with a variety of horrific themes and elements, featuring a killer werewolf, and is as much a horror film as it is a peplum.
</item 3>

<item 4>
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
</item 4>

<item 5>
Body Count (1987 film)
Body Count (released in Italy as Camping del Terrore/ Camping Terror) is a 1986 slasher film directed by Ruggero Deodato.

 It was released in Germany as "Body Count: Die Mathematik des Schreckens", and in Denmark as "Shamen".
</item 5>

<item 6>
Ruggero Deodato
Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
</item 6>

<item 7>
Cannibal Holocaust
Cannibal Holocaust is a 1980 Italian cannibal horror film directed by Ruggero Deodato from a screenplay by Gianfranco Clerici.

 It stars Carl Gabriel Yorke, Robert Kerman, Francesca Ciardi, Perry Pirkanen, and Luca Barbareschi.

 Influenced by the works of Mondo director Gualtiero Jacopetti, the film was inspired by Italian media reporting of Red Brigade terrorism.

 The coverage included news reports Deodato believed to be staged, an idea which became an integral aspect of the film's story.

 "Cannibal Holocaust" was filmed primarily in the Amazon rainforest of Colombia with indigenous tribes interacting with American and Italian actors.
</item 7>

<item 8>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 8>

<item 9>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 9>

<item 10>
Concorde Affaire '79
Concorde Affaire '79 (1979) (Italian: "Affare Concorde" ), also known as The Concorde Affair, is an Italian action thriller directed by Ruggero Deodato and
written by Ernesto Gastaldi and Renzo Genta.

 Released in the same year as "The Concorde ... Airport '79" and featuring actor Joseph Cotten, who appeared in "Airport '77", the film was an attempt by
producers to take advantage of the success of the "Airport" film franchise of the 1970s.
</item 10>
=== END ITEM id=ea96693477c13c5b ===

=== ITEM id=35f4fd1f2d7db9d1 ===
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
Robert B. Atwood Building
The Robert B. Atwood Building is an office building located at 550 West 7th Avenue in Downtown Anchorage, Alaska.

 The building houses government offices for the State of Alaska.

 Standing at 20 stories and 81 m (265 ft), it is the second-tallest building in Alaska.

 The building was formerly known as the "Bank of America Center".

 Together with the slightly taller Conoco-Phillips Building, the skyscraper defines Anchorage's skyline.
</item 1>

<item 2>
San Antonio Crowne Plaza
The San Antonio Crowne Plaza is a 10 story hotel near the San Antonio Airport San Antonio, Texas, USA.

 Located 2.6 miles from the San Antonio International Airport, this contemporary hotel is also 7 miles from the San Antonio River Walk and 10 miles from the
Alamo.
</item 2>

<item 3>
Alutiiq Museum
The Alutiiq Museum or Alutiiq Museum and Archaeological Repository is a non-profit museum and cultural center dedicated to preserving and sharing the cultural
traditions of the Koniag Alutiiq branch of Sugpiaq ~ Alutiiq of the Alaska Native people.

 The museum is a state-of-the-art facility and is located on the first floor of the "Alutiiq Center" of Kodiak, Alaska.

 Alutiiq Museum is one of three museums in Kodiak.

 The museum is the seventh museum in Alaska and the second "tribal museum" in the United States to be accredited.

 The museum provides tours of its exhibits, laboratory and collections storage facilities to educational groups.

 The museum will accept materials relevant to the prehistoric, historic, and contemporary cultural history of the Native peoples who settled the Koniag Alutiiq
Nation.

 Such materials include, but are not limited to, archaeological, ethnological, photographic, film, audio, archival, and natural history specimens.
</item 3>

<item 4>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 4>

<item 5>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 5>

<item 6>
Yakutat Airport
Yakutat Airport (IATA: YAK, ICAO: PAYA, FAA LID: YAK) is a state owned, public use airport located three nautical miles (6 km) southeast of the central business
district of Yakutat, a city and borough in the U.S. state of Alaska which has no road access to the outside world.

 Airline service is subsidized by the Essential Air Service program.
</item 6>

<item 7>
San Antonio International
San Antonio International was an American soccer club based in San Antonio, Texas that was a member of the Lone Star Soccer Alliance.
</item 7>

<item 8>
M7 Aerospace
M7 Aerospace LP is an aerospace company with its headquarters on the property of San Antonio International Airport in Uptown San Antonio, Texas, United States.
</item 8>

<item 9>
SyberJet Aircraft
SyberJet Aircraft is the manufacturer of the SJ30 light business jet.

 The company's headquarters is in Cedar City, Utah adjacent to the Cedar City Regional Airport with additional engineering offices and manufacturing, service,
repair and fatigue test facilities near and on the San Antonio International Airport in San Antonio, Texas.
</item 9>

<item 10>
Gates of the Arctic National Park and Preserve
Gates of the Arctic National Park and Preserve is a U.S. National Park in Alaska.

 It is the northernmost national park in the U.S. (the entirety of the park lies north of the Arctic Circle) and the second largest at 8,472,506 acres ,
slightly larger in area than Belgium.

 The park consists primarily of portions of the Brooks Range of mountains.

 It was first protected as a U.S. National Monument on December 1, 1978, before becoming a national park and preserve two years later in 1980 upon passage of
the Alaska National Interest Lands Conservation Act.

 A large part of the park is protected in the Gates of the Arctic Wilderness which covers 7,167,192 acre .

 The wilderness area adjoins the Noatak Wilderness Area and together they form the largest contiguous wilderness in the United States.
</item 10>
=== END ITEM id=35f4fd1f2d7db9d1 ===

=== ITEM id=b215327cca907c07 ===
Gold sentences and facts:
Sentence 0:  He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.
  Fact 0.0: The person finished second in voting for the Heisman Trophy in 1956.
  Fact 0.1: Paul Hornung finished first in that Heisman Trophy voting in 1956.
Sentence 1: Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.
  Fact 1.0: Paul Vernon Hornung was born on December 23, 1935.
  Fact 1.1: Paul Vernon Hornung is nicknamed "The Golden Boy".
  Fact 1.2: Paul Vernon Hornung is a former professional American football player.
  Fact 1.3: Paul Vernon Hornung is a Hall of Fame running back.
  Fact 1.4: Paul Vernon Hornung played for the Green Bay Packers of the National Football League (NFL).
  Fact 1.5: Paul Vernon Hornung played for the Green Bay Packers from 1957 to 1966.
  Fact 1.6: Paul Vernon Hornung won four NFL titles.
  Fact 1.7: Paul Vernon Hornung won the first Super Bowl.

Retrieved items:
<item 1>
1976 NCAA Division I football season
The 1976 NCAA Division I football season ended with a championship for the Panthers of the University of Pittsburgh.

 Led by head coach Johnny Majors (voted the AFCA Coach of the Year), the Pitt Panthers brought a college football championship to the home of the defending pro
football champions, the Steelers.

 Pitt also had the Heisman Trophy winner, Tony Dorsett; the Panthers had been ranked ninth in the preseason AP poll.
</item 1>

<item 2>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 2>

<item 3>
Frank Eliscu
Frank Eliscu (1912–1996) was an American sculptor and art teacher who designed and created the Heisman Memorial Football Trophy in 1935 when he was only 20
years old.

 The first Heisman Trophy, a strong young bull of a football player cast in bronze, was presented to a college football player in 1935, and is considered one of
the greatest honors a college athlete can receive.

 Over the years the Heisman Committee has paid tribute to Eliscu and his creation several times, and in 1985, the Heisman Committee invited him to speak at the
50th Anniversary Heisman ceremonies.
</item 3>

<item 4>
Paul Hornung
Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the
Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.

 He is the first in pro football history to win the Heisman Trophy, be selected as the first overall selection in the NFL Draft, win the NFL most valuable
player award, and be inducted into both the professional and college football halls of fame.
</item 4>

<item 5>
Charlie Ward
Charlie Ward Jr. (born October 12, 1970) is a retired American professional NBA basketball player, college football Heisman Trophy winner, Davey O'Brien Award
winner and a Major League Baseball draftee.

 Despite his NCAA football success, Ward was one of the very few players who won a Heisman trophy but was not drafted in the NFL draft.

 He won the College Football National Championship with the Florida State University Seminoles.

 Ward played several years with the New York Knicks and started in the NBA Finals.

 He was inducted in the College Football Hall of Fame in 2006.

 An avid tennis player, Ward also displayed his skills at the Arthur Ashe Tennis Tournament in 1994.
</item 5>

<item 6>
Johnny Majors
John Terrill Majors (born May 21, 1935) is a former American football player and coach.

 A standout halfback at the University of Tennessee, he was an All-American in 1956 and a two-time winner of the Southeastern Conference Most Valuable Player
award, in 1955 and 1956.

 He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.

 Majors served as the head football coach at Iowa State University (1968–1972), the University of Pittsburgh (1973–1976, 1993–1996), and Tennessee (1977–1992),
compiling a career college football record of 185–137–10.

 His 1976 Pittsburgh squad won a national championship after capping a 12–0 season with a victory in the Sugar Bowl.

 Majors was inducted into the College Football Hall of Fame as a player in 1987.
</item 6>

<item 7>
Terry Baker
Terry Wayne Baker (born May 5, 1941) is a former American football and basketball player.

 He played college football and college basketball at the Oregon State University.

 He played as a quarterback for the Oregon State Beavers football team from 1960 to 1962, winning the Heisman Trophy as senior.

 In the spring of his senior year, he played in the Final Four of the 1963 NCAA Men's Division I Basketball Tournament with the Oregon State Beavers men's
basketball team.

 To date, he is the only athlete to win a Heisman Trophy and play in the Final Four.

 Baker was the first overall pick in the 1963 NFL draft and played with the Los Angeles Rams of the National Football League (NFL) from 1963 to 1965.

 He then played for one season in the Canadian Football League (CFL) with the Edmonton Eskimos, in 1966.

 Baker was inducted into the College Football Hall of Fame in 1982.
</item 7>

<item 8>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 8>

<item 9>
Heisman Trophy
The Heisman Memorial Trophy (usually known colloquially as the Heisman Trophy or The Heisman), is awarded annually to the most outstanding player in college
football in the United States whose performance best exhibits the pursuit of excellence with integrity.

 Winners epitomize great ability combined with diligence, perseverance, and hard work.

 It is presented by the Heisman Trophy Trust in early December before the postseason bowl games.
</item 9>

<item 10>
1997 Tennessee Volunteers football team
However, the Vols' defense could not stop Nebraska's rushing attack, giving up more than 400 yards on the ground in a 42–17 loss.

 As a senior, Manning won numerous awards.

 He was a consensus first-team All-American and won the Maxwell Award, the Davey O'Brien Award, the Johnny Unitas Award, and the Best College Football Player
ESPY Award, among others.

 However, he did not win the Heisman Trophy, finishing runner-up to Charles Woodson, a CB from Michigan, and the only defensive player ever to win the Heisman
Trophy.
</item 10>
=== END ITEM id=b215327cca907c07 ===

=== ITEM id=d3f5229f8e664434 ===
Gold sentences and facts:
Sentence 0: Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.
  Fact 0.0: Jürgen Melzer was born on 22 May 1981.
  Fact 0.1: Jürgen Melzer was born in Vienna.
  Fact 0.2: Jürgen Melzer is an Austrian tennis player.
Sentence 1: Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
  Fact 1.0: Tom Gullikson was born on September 8, 1951.
  Fact 1.1: Tom Gullikson is a tennis coach.
  Fact 1.2: Tom Gullikson is a former professional tennis player.
  Fact 1.3: Tom Gullikson is from the United States.

Retrieved items:
<item 1>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 1>

<item 2>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 2>

<item 3>
Julian Knowle
Julian Knowle (born 29 April 1974) is an Austrian male professional tennis player.

 Being a born left-hander, Knowle is now one of the few on the ATP Tour who plays his forehand, backhand, and even volleys double-handed.

 He was Austria's most successful doubles player in history by reaching world no. 6 in the ATP doubles rankings in January 2008, before being matched by Jürgen
Melzer, who reached no. 6 in September 2010, and overtaken by Alexander Peya, who reached no. 3 in August 2013.
</item 3>

<item 4>
2010 Pilot Pen Tennis – Men's Doubles
Julian Knowle and Jürgen Melzer were the defending champions, but Melzer chose not to participate this year.As a result, Knowle partnered with Andy Ram, but
they lost to Robert Lindstedt and Horia Tecău in the semifinals.

 Lindstedt and Tecău went on to win the tournament, after defeating Rohan Bopanna and Aisam-ul-Haq Qureshi in the final 6–4, 7–5.
</item 4>

<item 5>
2012 US Open – Men's Doubles
Jürgen Melzer and Philipp Petzschner were the defending champions, but lost in the second round to Jamie Delgado and Ken Skupski.

 Melzer and Petzschner needed a wildcard to compete because they forgot to sign up.

<br>
</item 5>

<item 6>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 6>

<item 7>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 7>

<item 8>
Ernests Gulbis
Ernests Gulbis (] , born 30 August 1988, and nicknamed "The Gull" or "Ernie") is a Latvian professional tennis player.

 In 2008, Gulbis won his first ATP Tour doubles title at the U.S. Men's Clay Court Championships, teaming with Rainer Schüttler, and in 2010 won his first ATP
Tour singles title in the Delray Beach, defeating Ivo Karlović in the final.

 In total, Gulbis has 6 ATP titles to his name and has never lost a final.

 His best performance at a Grand Slam is reaching the semifinals of the 2014 French Open.

 He had previously reached the quarterfinals of the 2008 French Open.

 Gulbis' career-high singles ranking is world No. 10, making him the only Latvian tennis player ever to be ranked inside the top 10 in ATP Singles Ranking.

 He achieved this in June 2014.
</item 8>

<item 9>
Tom Gullikson
Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
</item 9>

<item 10>
1983 Bristol Open – Doubles
Tim Gullikson and Tom Gullikson were the defending champions, but Tim Gullikson did not participate this year.

 Tom Gullikson partnered Johan Kriek, losing in the final.
</item 10>
=== END ITEM id=d3f5229f8e664434 ===

=== ITEM id=8c4406cc73c7cd3a ===
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
Fred Winter
Frederick Thomas Winter CBE (20 September 1926 – 5 April 2004) was a British National Hunt racing racehorse jockey and trainer.

 He was British jump racing Champion Jockey four times and British jump racing Champion Trainer eight times.

 He is the only person to have won the Cheltenham Gold Cup, Champion Hurdle and Grand National as both jockey and trainer.

 Winter won the Grand National four times, as a jockey in 1957 (Sundew) and 1962 (Kilmore), and as a trainer in 1965 (Jay Trump) and 1966 (Anglo).
</item 1>

<item 2>
Bart Cummings
James Bartholomew Cummings {'1': ", '2': ", '3': ", '4': "} (14 November 1927 – 30 August 2015), also known by his initials J. B. Cummings, was one of the most
successful Australian racehorse trainers.

 He was known as the "Cups King", referring to the Melbourne Cup, as he won 'the race that stops a nation' a record twelve times.

 During his lifetime Cummings was considered an Australian cultural icon and an Australian National Living Treasure.
</item 2>

<item 3>
Tony McCoy
Sir Anthony Peter McCoy, OBE (born 4 May 1974), commonly known as A. P. McCoy or Tony McCoy, is a Northern Irish former horse racing jockey.

 Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.

 He stands 1.78 m (5'10"), far taller than most jockeys.
</item 3>

<item 4>
Binocular (horse)
Binocular, foaled on March 17, 2004 in France, is a French thoroughbred retired racehorse, born to sire, Enrique and dam Bleu Ciel Et Blanc.

 He is owned by J. P. McManus and trained by Nicky Henderson.

 His primary jockey is Tony McCoy.
</item 4>

<item 5>
Irish flat racing Champion Jockey
The Champion Jockey of flat racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1950.
</item 5>

<item 6>
Irish jump racing Champion Jockey
The Champion Jockey of National Hunt racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1947.

 The current champion, as of 2017, is Ruby Walsh.
</item 6>

<item 7>
Gordon Richards
Sir Gordon Richards (5 May 1904 – 10 November 1986) was an English jockey.

 He was the British flat racing Champion Jockey 26 times and is often considered the world's greatest ever jockey.

 He remains the only flat jockey to have been knighted.
</item 7>

<item 8>
British flat racing Champion Jockey
The Champion Jockey of flat racing in Great Britain is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey and the number of winners for each year since 1840.

 The seasonal record of jockeys' winners was published for the first time in 1846.

 The championship was sponsored for the first time in 2009 by online casino 32Red and is currently sponsored by Stobart Group.

 The 2016 title was determined by the number of winners ridden between 30 April and 15 October, in both turf and all-weather racing.

 Prior to 2015 the title was decided over the full length of the Flat turf season, from March to November, and carried no prize money to the winner.

 A prize of £25,000 to the champion jockey was also introduced as part of the 2015 changes.
</item 8>

<item 9>
German flat racing Champion Jockey
The Champion Jockey of flat racing in Germany is the jockey who has ridden the most winning horses during a season.

 The list of the Champion Jockey started 1870.

 The list below shows the Champion Jockey for each year since 1984.
</item 9>

<item 10>
Buster Parnell
Ryan Farnham "Buster" Parnell (c.1934 – 17 September 2017) was an Irish jockey who competed in Flat racing.

 Parnell was Irish flat racing Champion Jockey in 1969 and won three Irish Classic Races.

 He was born in Fulham and rode his first winner in 1950.

 He subsequently served in the Royal Air Force before returning to horse racing and becoming champion jockey in Denmark.

 He spent the majority of his career in Ireland.

 His son, David Parnell, was also a jockey.
</item 10>
=== END ITEM id=8c4406cc73c7cd3a ===

=== ITEM id=c5d74acf0cf6b536 ===
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
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
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
Treasure Island Development
The Treasure Island Development is a 405 acre major redevelopment project under construction on Treasure Island and parts of Yerba Buena Island in San Francisco
Bay between San Francisco and Oakland, within San Francisco city limits.

 The Treasure Island Development Authority (TIDA) is a nonprofit organization formed to oversee the economic development of the former naval station.

 Treasure Island's development was set to break ground during mid-2012.

 However, on April 12, 2013, The San Francisco Chronicle reported that the deal has collapsed, with the Chinese investors from China Development Bank and China
Railway Construction Corporation withdrawing from the project.

 The Treasure Island Project is now being developed by a joint venture between Lennar Corporation and Kenwood Investments.

 The development is expected to cost US$ .
</item 6>

<item 7>
Rinku Singh
Rinku Singh (born August 8, 1988 in Lucknow, India) is a left-handed baseball pitcher who is currently a free agent.

 Singh was signed by the Pittsburgh Pirates organization after he won a pitching contest on a 2008 reality television show, "The Million Dollar Arm".

 He was the first Indian to play professional baseball and has spent several seasons in the minor leagues, only reaching as high as the A level.

 He is the subject of the movie "Million Dollar Arm".
</item 7>

<item 8>
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
</item 8>

<item 9>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
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
=== END ITEM id=c5d74acf0cf6b536 ===

=== ITEM id=17ecd8dfa2172fbc ===
Gold sentences and facts:
Sentence 0: Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.
  Fact 0.0: Jürgen Melzer was born on 22 May 1981.
  Fact 0.1: Jürgen Melzer was born in Vienna.
  Fact 0.2: Jürgen Melzer is an Austrian tennis player.
Sentence 1: Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
  Fact 1.0: Tom Gullikson was born on September 8, 1951.
  Fact 1.1: Tom Gullikson is a tennis coach.
  Fact 1.2: Tom Gullikson is a former professional tennis player.
  Fact 1.3: Tom Gullikson is from the United States.

Retrieved items:
<item 1>
2010 Pilot Pen Tennis – Men's Doubles
Julian Knowle and Jürgen Melzer were the defending champions, but Melzer chose not to participate this year.As a result, Knowle partnered with Andy Ram, but
they lost to Robert Lindstedt and Horia Tecău in the semifinals.

 Lindstedt and Tecău went on to win the tournament, after defeating Rohan Bopanna and Aisam-ul-Haq Qureshi in the final 6–4, 7–5.
</item 1>

<item 2>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 2>

<item 3>
2012 US Open – Men's Doubles
Jürgen Melzer and Philipp Petzschner were the defending champions, but lost in the second round to Jamie Delgado and Ken Skupski.

 Melzer and Petzschner needed a wildcard to compete because they forgot to sign up.

<br>
</item 3>

<item 4>
Ernests Gulbis
Ernests Gulbis (] , born 30 August 1988, and nicknamed "The Gull" or "Ernie") is a Latvian professional tennis player.

 In 2008, Gulbis won his first ATP Tour doubles title at the U.S. Men's Clay Court Championships, teaming with Rainer Schüttler, and in 2010 won his first ATP
Tour singles title in the Delray Beach, defeating Ivo Karlović in the final.

 In total, Gulbis has 6 ATP titles to his name and has never lost a final.

 His best performance at a Grand Slam is reaching the semifinals of the 2014 French Open.

 He had previously reached the quarterfinals of the 2008 French Open.

 Gulbis' career-high singles ranking is world No. 10, making him the only Latvian tennis player ever to be ranked inside the top 10 in ATP Singles Ranking.

 He achieved this in June 2014.
</item 4>

<item 5>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 5>

<item 6>
1983 Bristol Open – Doubles
Tim Gullikson and Tom Gullikson were the defending champions, but Tim Gullikson did not participate this year.

 Tom Gullikson partnered Johan Kriek, losing in the final.
</item 6>

<item 7>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 7>

<item 8>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 8>

<item 9>
Julian Knowle
Julian Knowle (born 29 April 1974) is an Austrian male professional tennis player.

 Being a born left-hander, Knowle is now one of the few on the ATP Tour who plays his forehand, backhand, and even volleys double-handed.

 He was Austria's most successful doubles player in history by reaching world no. 6 in the ATP doubles rankings in January 2008, before being matched by Jürgen
Melzer, who reached no. 6 in September 2010, and overtaken by Alexander Peya, who reached no. 3 in August 2013.
</item 9>

<item 10>
Tom Gullikson
Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
</item 10>
=== END ITEM id=17ecd8dfa2172fbc ===
