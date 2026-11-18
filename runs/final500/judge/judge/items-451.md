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

This file has 13 items.

=== ITEM id=8afc56689f25fa33 ===
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
Sam Eagle
Sam Eagle is a Muppet character originating from the television show "The Muppet Show", where he was performed by Frank Oz.

 Sam has appeared in every Muppet film; as himself in "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "Muppets from Space", and "The
Muppets", as well as the Head Schoolmaster in "The Muppet Christmas Carol," Samuel Arrow in "Muppet Treasure Island" and a CIA agent in "Muppets Most Wanted."

 He also appears in the television series, "The Muppets."
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
</item 3>

<item 4>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
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
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 6>

<item 7>
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 7>

<item 8>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 8>

<item 9>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 9>

<item 10>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 10>
=== END ITEM id=8afc56689f25fa33 ===

=== ITEM id=b41d02d8f3df1d99 ===
Gold sentences and facts:
Sentence 0: Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.
  Fact 0.0: Nelson Algren was born March 28, 1909.
  Fact 0.1: Nelson Algren died May 9, 1981.
  Fact 0.2: Nelson Algren was an American writer.
Sentence 1: Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
  Fact 1.0: Nathanael West was born Nathan Weinstein.
  Fact 1.1: Nathanael West was born October 17, 1903.
  Fact 1.2: Nathanael West died December 22, 1940.
  Fact 1.3: Nathanael West was an American author.
  Fact 1.4: Nathanael West was an American screenwriter.
  Fact 1.5: Nathanael West was an American satirist.

Retrieved items:
<item 1>
Haniel Long
Haniel Clark Long (March 9, 1888 – October 17, 1956) was an American poet, novelist, publisher and academic.

 He is best known for his novella, "Interlinear to Cabeza de Vaca" (1936), a fictionalized account of the true story of a Spanish conquistador in 16th century
North America.
</item 1>

<item 2>
Nelson (band)
Nelson is an American rock band founded by singer/songwriters Matthew and Gunnar Nelson (twin sons of Ricky Nelson and Kristin Nelson).

 The band achieved success during the early 1990s with their double platinum debut album "After the Rain", which featured the number-one hit "(Can't Live
Without Your) Love and Affection".
</item 2>

<item 3>
Andrew Glaze
Andrew Glaze (April 21, 1920 – February 7, 2016) was an American poet, playwright and novelist.

 Much of Glaze's poetry reflects his coming of age in the South, and his eventual return there.

 He also lived and wrote in New York City for 31 years.

 In New York City he became part of a circle of poets that included Oscar Williams, Norman Rosten, John Ciardi and William Packard.
</item 3>

<item 4>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 4>

<item 5>
Nathanael West
Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
</item 5>

<item 6>
The Man with the Golden Arm (novel)
The Man with the Golden Arm is a novel by Nelson Algren, published by Doubleday in November 1949.

 One of the seminal novels of post-World War II American letters, "The Man with the Golden Arm" is widely considered Algren's greatest and most enduring work.

 It won the National Book Award in 1950.
</item 6>

<item 7>
Romeo Nelson
Iromeio "Romeo" Nelson (March 12, 1902 – May 17, 1974) was an American boogie woogie pianist whose recordings from 1929 are regarded as some of the finest, and
certainly the fastest, boogie woogie showpieces on record.
</item 7>

<item 8>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 8>

<item 9>
A Walk on the Wild Side
A Walk on the Wild Side is a 1956 novel by Nelson Algren, most often quoted as the source for Algren's "three rules of life": "Never play cards with a man
called Doc.

 Never eat at a place called Mom's.

 Never sleep with a woman whose troubles are worse than your own."
</item 9>

<item 10>
Gunnar Nelson (musician)
Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
</item 10>
=== END ITEM id=b41d02d8f3df1d99 ===

=== ITEM id=500840178e261841 ===
Gold sentences and facts:
Sentence 0: Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
  Fact 0.0: Distortion Mirrors is an English alternative rock band.
  Fact 0.1: Distortion Mirrors is based in Leeds, West Yorkshire, England.
  Fact 0.2: Distortion Mirrors was formed in 2011.
Sentence 1: OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.
  Fact 1.0: OK Go is an American rock band.
  Fact 1.1: OK Go was originally from Chicago, Illinois.
  Fact 1.2: OK Go is now based in Los Angeles, California.

Retrieved items:
<item 1>
Twelve Days of OK Go
Twelve Days of OK Go is a compilation album by American rock band OK Go.

 It was released on December 31, 2012.

 OK Go started releasing the songs on December 10, with one song released each weekday.

 The last song, a cover of "Any Time at All", was released on Christmas.

 A bonus track, a cover of "This Will Be Our Year," was released on New Year's Eve.
</item 1>

<item 2>
The Writing's on the Wall (OK Go song)
"The Writing's on the Wall" is a song by American rock band OK Go.

 It was released on June 17, 2014, as part of the band's EP "Upside Out", and is also the first single from the band's fourth studio album "Hungry Ghosts".

 On the same day, the band released a music video in which the members use props to create optical illusions, reflecting the song's description of a
relationship that fails because the couple has different points of view.

 Like previous OK Go videos, it is structured as a one-shot music video.

 The many YouTube views of the video caused the song to debut in the top ten of the US "Billboard" Hot Rock Songs chart, as well as number one on the Bubbling
Under Hot 100 Singles chart.
</item 2>

<item 3>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 3>

<item 4>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 4>

<item 5>
OK Go videography
The musical rock band OK Go has earned considerable fame for their creative but often low-budget music videos, most of which have been promoted through Internet
video sharing sites like YouTube.

 Many of these have become viral videos; the 2006 video for "Here It Goes Again", in which the band performed a complex routine with the aid of motorized
treadmills, has received over 50 million views four years later.

 Their video for Needing/Getting, released February 5, 2012 in partnership with Chevrolet, debuted during Super Bowl XLVI and has over 32 million views on
YouTube.

 Samuel Bayer, who produced many music videos in the 1990s, asserted that OK Go's promotion of music videos on the Internet was akin to Nirvana's ushering in
the grunge movement.

 Many of the videos also use long or single-shot takes, which "Salon"' s Matt Zoller Seitz says "restore[s] a sense of wonder to the musical number by letting
the performers' humanity shine through and allowing them to do their thing with a minimum of filmmaking interference".
</item 5>

<item 6>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 6>

<item 7>
OK Go (album)
OK Go is the debut studio album by American rock band OK Go.

 It was released in September 2002.

 The cover was created by designer Stefan Sagmeister.
</item 7>

<item 8>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 8>

<item 9>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 9>

<item 10>
OK Go videography
The success of OK Go's music first won the band the 14th Annual Webby Special Achievement Award for Film and Video Artist of the Year.

 The video for "This Too Shall Pass" was named both "Video of the Year" and "Best Rock Video" at the 3rd annual UK Music Video Awards.

"This Too Shall Pass" won the LA Film Fest's Audience Award for Best Music Video, UK MVA Awards – Music Video of the Year Winner 2010, among others.
</item 10>
=== END ITEM id=500840178e261841 ===

=== ITEM id=aba4112964f2ff8a ===
Gold sentences and facts:
Sentence 0:  His siblings were Abby, Nelson, Laurance, Winthrop, and David.
  Fact 0.0: Abby was his sibling.
  Fact 0.1: Nelson was his sibling.
  Fact 0.2: Laurance was his sibling.
  Fact 0.3: Winthrop was his sibling.
  Fact 0.4: David was his sibling.
Sentence 1: David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.
  Fact 1.0: David Rockefeller lived from June 12, 1915 to March 20, 2017.
  Fact 1.1: David Rockefeller was an American banker.
  Fact 1.2: David Rockefeller was chairman of Chase Manhattan Corporation.
  Fact 1.3: David Rockefeller was chief executive of Chase Manhattan Corporation.

Retrieved items:
<item 1>
JPMorgan Chase
JPMorgan Chase & Co. is a U.S. multinational banking and financial services holding company headquartered in New York City.

 It is the largest bank in the United States, the world's sixth largest bank by total assets, with total assets of US$2.5 trillion, and the world's second most
valuable bank by market capitalization, after ICBC.

 It is a major provider of financial services, and according to "Forbes" magazine is the world's sixth largest public company based upon a composite ranking.

 The hedge fund unit of JPMorgan Chase is the second largest hedge fund in the United States.

 The company was formed in 2000, when Chase Manhattan Corporation merged with J.P. Morgan & Co.
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
John D. Rockefeller III
John Davison Rockefeller III (March 21, 1906 – July 10, 1978) was a philanthropist and third-generation member of the prominent Rockefeller family.

 He was the eldest son of philanthropists John D. Rockefeller Jr. and Abby Aldrich Rockefeller.

 His siblings were Abby, Nelson, Laurance, Winthrop, and David.
</item 3>

<item 4>
David Rockefeller
David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.

 He was the oldest living member of the Rockefeller family and family patriarch from August 2004 until his death in March 2017.

 Rockefeller was a son of John D. Rockefeller Jr. and Abby Aldrich Rockefeller, and a grandson of John D. Rockefeller and Laura Spelman Rockefeller.
</item 4>

<item 5>
Rockefeller family
The Rockefeller family ( ) is an American industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the
late 19th and early 20th centuries, with John D. Rockefeller and his brother William Rockefeller primarily through Standard Oil.

 The family is also known for its long association with and control of Chase Manhattan Bank.

 They are considered to be one of the most powerful families, if not the most powerful family, in the history of the United States.
</item 5>

<item 6>
William B. Harrison Jr.
William B. Harrison Jr., born August 12, 1943, in Rocky Mount, North Carolina, is the former CEO and chairman of JPMorgan Chase.

 He attended high school at Virginia Episcopal School, where he was a basketball star.

 He attended the University of North Carolina at Chapel Hill, where he was admitted to the Zeta Psi fraternity.

 Having risen through the ranks of Chemical Bank before succeeding Walter V. Shipley during the Chemical Bank takeover of the Chase Manhattan Corporation, he
and Douglas A. Warner III, then CEO of J.P. Morgan & Co., were the principal architects of the US$30.9 billion Chase and J.P. Morgan & Co. merger of 2000.

 Harrison has been a director of the Firm or a predecessor institution since 1991.

 Harrison is also a director of Merck & Co., Inc.
</item 6>

<item 7>
Chase Bank
JPMorgan Chase Bank, N.A., doing business as Chase Bank, is a national bank that constitutes the consumer and commercial banking subsidiary of the U.S.
multinational banking and financial services holding company, JPMorgan Chase.

 The bank was known as Chase Manhattan Bank until it merged with J.P. Morgan & Co. in 2000.

 Chase Manhattan Bank was formed by the merger of the Chase National Bank and The Manhattan Company in 1955.

 The bank has been headquartered in Columbus, Ohio since its merger with Bank One Corporation in 2004.

 The bank acquired the deposits and most assets of Washington Mutual.
</item 7>

<item 8>
Hope Aldrich Rockefeller
Hope Aldrich Rockefeller (born May 17, 1938) is an American newspaper publisher.

 She is the eldest daughter of philanthropist John Davison Rockefeller III (1906–1978) and Blanchette Ferry Hooker (1909–1992).

 Her elder brother is former Senator John Davison "Jay" Rockefeller IV (born 1937).
</item 8>

<item 9>
Rockefeller Brothers Fund
The Rockefeller Brothers Fund (RBF) is a philanthropic foundation created and run by members of the Rockefeller family.

 It was founded in New York City in 1940 as the primary philanthropic vehicle for the five third-generation Rockefeller brothers: John D. Rockefeller III,
Nelson, Laurance, Winthrop and David.

 It is distinct from the Rockefeller Foundation.

 The Rockefellers are an industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the late 19th and
early 20th centuries.
</item 9>

<item 10>
Walter V. Shipley
Walter V. Shipley (born November 2, 1935) was the former Chairman and Chief Executive Officer of Chase Manhattan Bank and, before that, the company with which
it merged Chemical Bank.

 Shipley was named chief executive of Chemical in 1981 and held the position through 1999 and remained at the bank as chairman through January 2000, just prior
to the bank's merger with J.P. Morgan & Co..

 During his 18-year tenure, Shipley oversaw Chemical's mergers with Texas Commerce Bank in 1987, Manufacturers Hanover in 1991 and Chase Manhattan Bank in 1996.
</item 10>
=== END ITEM id=aba4112964f2ff8a ===

=== ITEM id=a3181299c1878580 ===
Gold sentences and facts:
Sentence 0: Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
  Fact 0.0: Mauritz Stiller was born on 17 July 1883.
  Fact 0.1: Mauritz Stiller died on 18 November 1928.
  Fact 0.2: Mauritz Stiller was a Finnish-Swedish film director.
  Fact 0.3: Mauritz Stiller was best known for discovering Greta Garbo.
  Fact 0.4: Mauritz Stiller brought Greta Garbo to America.
Sentence 1: Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.
  Fact 1.0: Johannes Grenzfurthner was born in 1975 in Vienna.
  Fact 1.1: Johannes Grenzfurthner is an award-winning Austrian artist.
  Fact 1.2: Johannes Grenzfurthner is an Austrian filmmaker.
  Fact 1.3: Johannes Grenzfurthner is an Austrian writer.
  Fact 1.4: Johannes Grenzfurthner is an Austrian actor.
  Fact 1.5: Johannes Grenzfurthner is an Austrian curator.
  Fact 1.6: Johannes Grenzfurthner is an Austrian theatre director.
  Fact 1.7: Johannes Grenzfurthner is an Austrian lecturer.

Retrieved items:
<item 1>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 1>

<item 2>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 2>

<item 3>
Johannes Grenzfurthner
Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.

 He is known as the founder, conceptualist and artistic director of monochrom, an international art and theory group.

 Most of his artworks are labelled "monochrom".
</item 3>

<item 4>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 4>

<item 5>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 5>

<item 6>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 6>

<item 7>
Mauritz Stiller
Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
</item 7>

<item 8>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 8>

<item 9>
Richard Genée
Franz Friedrich Richard Genée (February 7, 1823 – June 15, 1895) was a Prussian born Austrian librettist, playwright, and composer.

 He is most famous for the libretto of "Die Fledermaus", Johann Strauss II's most famous operetta.

 He co-wrote the libretto without having met top-billed librettist Karl Haffner, who constructed the new story based on a play by Henri Meilhac and Ludovic
Halévy, which was considered too shocking to perform outside Paris.

 Genée, however, wrote the operetta's actual text and drew nothing from Haffner beyond the names of the characters.
</item 9>

<item 10>
Traceroute (film)
Traceroute is a 2016 Austrian/American documentary film directed by Johannes Grenzfurthner.

 The autobiographical documentary and road movie deals with the history, politics and impact of nerd culture.

 Grenzfurthner calls his film a "personal journey into the uncharted depths of nerd culture, a realm full of dangers, creatures and more or less precarious
working conditions", an attempt to "chase the ghosts of nerddom's past, present and future."

 The film was co-produced by art group monochrom and Reisenbauer Film.

 It features music by Kasson Crooker, Hans Nieswandt, and many others.
</item 10>
=== END ITEM id=a3181299c1878580 ===

=== ITEM id=09ac5119d2ceb5d4 ===
Gold sentences and facts:
Sentence 0: Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space as part of the Soviet Union's Interkosmos programme.
  Fact 0.0: Sigmund Werner Paul Jähn was born on 13 February 1937.
  Fact 0.1: Sigmund Werner Paul Jähn is a German cosmonaut.
  Fact 0.2: Sigmund Werner Paul Jähn is a pilot.
  Fact 0.3: In 1978 Sigmund Werner Paul Jähn became the first East German to fly in space.
  Fact 0.4: In 1978 Sigmund Werner Paul Jähn became the first German native to fly in space.
  Fact 0.5: The 1978 flight by Sigmund Werner Paul Jähn was part of the Soviet Union's Interkosmos programme.
Sentence 1: Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.
  Fact 1.0: Arne Christer Fuglesang was born on March 18, 1957.
  Fact 1.1: Arne Christer Fuglesang was born in Stockholm.
  Fact 1.2: Arne Christer Fuglesang is a Swedish physicist.
  Fact 1.3: Arne Christer Fuglesang is an ESA astronaut.
Sentence 2:  He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
  Fact 2.0: Arne Christer Fuglesang was first launched aboard the STS-116 Space Shuttle mission.
  Fact 2.1: That launch was on December 10, 2006, at 01:47 GMT.
  Fact 2.2: Arne Christer Fuglesang became the first Swedish citizen in space.

Retrieved items:
<item 1>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 1>

<item 2>
Samantha Cristoforetti
Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.

 She holds the record for the longest uninterrupted spaceflight of a European astronaut (199 days, 16 hours), and until June 2017 held the record for the
longest single space flight by a woman until this was broken by Peggy Whitson.

 She is also the first Italian woman in space.

 Samantha Cristoforetti is also known as the first person who brewed an espresso coffee in space.
</item 2>

<item 3>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 3>

<item 4>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 4>

<item 5>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 5>

<item 6>
Eberhard Köllner
Eberhard Köllner (born 29 September 1939 in Stassfurt, Germany) was selected for Soyuz 31 as the backup for Sigmund Jähn.
</item 6>

<item 7>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 7>

<item 8>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 8>

<item 9>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 9>

<item 10>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 10>
=== END ITEM id=09ac5119d2ceb5d4 ===

=== ITEM id=a2cc5b073b8b45f3 ===
Gold sentences and facts:
Sentence 0: Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
  Fact 0.0: Scott Treleaven is a Canadian artist.
  Fact 0.1: Scott Treleaven's work employs a variety of media.
  Fact 0.2: The media employed in Scott Treleaven's work include collage, film, video, drawing, photography and installation.
Sentence 1: Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.
  Fact 1.0: Edmund Mortimer's dates are 21 August 1874 to 21 May 1944.
  Fact 1.1: Edmund Mortimer was an American actor.
  Fact 1.2: Edmund Mortimer was an American film director.

Retrieved items:
<item 1>
Edmund Mortimer, 2nd Baron Mortimer
Edmund Mortimer, 2nd Lord Mortimer (1251 – 17 July 1304) was the second son and eventual heir of Roger Mortimer, 1st Baron Mortimer.

 His mother was Maud de Braose.

 As a younger son, Edmund had been intended for clerical or monastic life, and had been sent to study at Oxford University.
</item 1>

<item 2>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 2>

<item 3>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 3>

<item 4>
Edmund Mortimer, 5th Earl of March
Edmund de Mortimer, 5th Earl of March and 7th Earl of Ulster (6 November 1391 – 18 January 1425) was an English nobleman.

 A great-great-grandson of King Edward III of England, he was heir presumptive to King Richard II of England, his first cousin twice removed, when Richard II
was deposed in favour of Henry IV.

 Edmund Mortimer's claim to the throne was the basis of rebellions and plots against Henry IV and his son Henry V, and was later taken up by the House of York
in the Wars of the Roses, though Mortimer himself was an important and loyal vassal of Henry V and Henry VI.

 Edmund Mortimer was the last Earl of March of the Mortimer family.
</item 4>

<item 5>
Tommy Atkins (director)
Tommy Atkins was an American director of the silent and early sound film eras.

 Born on July 18, 1887 in Springfield, Massachusetts, he would make his entrance into the film industry as the assistant director to Ralph Ince on the 1920
silent film, "Out of the Snows".

 It would be another eight years before he would make another film, again as assistant director, this time for FBO Pictures, on another silent film, "Crooks
Can't Win".

 He'd work as the assistant director on another sixteen films between 1928 and 1934, the most notable of which would be 1933's "Morning Glory", directed by
Lowell Sherman and starring Katharine Hepburn and Douglas Fairbanks Jr..

 In 1934 he would be given the chance to helm his first picture, "The Silver Streak", which was one of the top money-makers for RKO Pictures that year.

 He would only direct two more films, the second of which, "Hi, Gaucho!

", he would also write the story for.
</item 5>

<item 6>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 6>

<item 7>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 7>

<item 8>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 8>

<item 9>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 9>

<item 10>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 10>
=== END ITEM id=a2cc5b073b8b45f3 ===

=== ITEM id=d98973fbf618b4be ===
Gold sentences and facts:
Sentence 0: The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.
  Fact 0.0: The Shame of a City is a 2006 feature-length documentary.
  Fact 0.1: The Shame of a City premiered at the Philadelphia Film Festival.
Sentence 1: Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and preserved in the WPA Slave Narrative Collection.
  Fact 1.0: Unchained Memories is a 2003 documentary film.
  Fact 1.1: Unchained Memories is about the stories of former slaves.
  Fact 1.2: The former slaves were interviewed during the 1930s.
  Fact 1.3: The interviews were part of the Federal Writers' Project.
  Fact 1.4: The stories are preserved in the WPA Slave Narrative Collection.

Retrieved items:
<item 1>
Unchained Memories
Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and
preserved in the WPA Slave Narrative Collection.

 This HBO film interpretation directed by Ed Bell and Thomas Lennon is a compilation of slave narratives, narrated by actors, emulating the original
conversation with the interviewer.

 The slave narratives may be the most accurate in terms of the everyday activities of the enslaved, serving as personal memoirs of more than two thousand former
slaves.

 The documentary depicts the emotions of the slaves and what they endured.

 The "Master" had the opportunity to sell, trade, or kill the enslaved, for retribution should one slave not obey.
</item 1>

<item 2>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 2>

<item 3>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 3>

<item 4>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 4>

<item 5>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 5>

<item 6>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 6>

<item 7>
The Shame of a City
The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.

 Filmmaker Tigre Hill chronicles the 2003 Philadelphia mayoral race between Democrat incumbent mayor John Street and Republican challenger Sam Katz.

 Early polls showed Katz with a small lead in this predominantly Democratic city but twenty-seven days before the election, an FBI bug was found in the mayor’s
office.

 The discovery at first seemed like a death knell to the Street campaign and a near certain victory for Katz.

 Yet this prediction was proven wrong when Street and his supporters successfully polarized the campaign by leveling accusations of instituational racial
prejudice and playing on historical skepticism of the Republican-controlled federal government.

 As a result, Street won re-election by a sixteen-point margin.
</item 7>

<item 8>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 8>

<item 9>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 9>

<item 10>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
</item 10>
=== END ITEM id=d98973fbf618b4be ===

=== ITEM id=3eea503fcf524d70 ===
Gold sentences and facts:
Sentence 0: Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.
  Fact 0.0: Sir Nicholas Robert Hytner was born on 7 May 1956.
  Fact 0.1: Sir Nicholas Robert Hytner is an English theatre director.
  Fact 0.2: Sir Nicholas Robert Hytner is an English film director.
  Fact 0.3: Sir Nicholas Robert Hytner is an English film producer.
Sentence 1: Maya Deren (April 29, 1917 – October 13, 1961), born Eleanora Derenkowskaia (Russian: Элеоно́ра Деренко́вская ), was a Russian-American filmmaker and one of the most important American experimental filmmakers and entrepreneurial promoters of the avant-garde in the 1940s and 1950s.
  Fact 1.0: Maya Deren was born on April 29, 1917.
  Fact 1.1: Maya Deren died on October 13, 1961.
  Fact 1.2: Maya Deren was born Eleanora Derenkowskaia.
  Fact 1.3: The birth name Eleanora Derenkowskaia is given in Russian as Элеоно́ра Деренко́вская.
  Fact 1.4: Maya Deren was a Russian-American filmmaker.
  Fact 1.5: Maya Deren was one of the most important American experimental filmmakers in the 1940s and 1950s.
  Fact 1.6: Maya Deren was one of the most important American entrepreneurial promoters of the avant-garde in the 1940s and 1950s.
Sentence 2:  Deren was also a choreographer, dancer, film theorist, poet, lecturer, writer and photographer.
  Fact 2.0: Maya Deren was also a choreographer.
  Fact 2.1: Maya Deren was also a dancer.
  Fact 2.2: Maya Deren was also a film theorist.
  Fact 2.3: Maya Deren was also a poet.
  Fact 2.4: Maya Deren was also a lecturer.
  Fact 2.5: Maya Deren was also a writer.
  Fact 2.6: Maya Deren was also a photographer.

Retrieved items:
<item 1>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 1>

<item 2>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 2>

<item 3>
Maya Deren Award
The American Film Institute Award for Independent Film and Video Artists, subtitled and generally known as the Maya Deren Award, was an award presented to
filmmakers and video artists by the American Film Institute to honor independent filmmaking.

 Named for the avant-garde experimental film artist Maya Deren, it was given from 1986 through 1996.
</item 3>

<item 4>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 4>

<item 5>
Tereza Vilišová
Tereza Vilišová (born 1981) is a Czech actress.

 She won the Alfréd Radok Award for Best Actress in 2013 for her role of "Amy" in the D C Jackson play "My Romantic History" at the Divadlo Petra Bezruče in
Ostrava.

 At the 2013 Thalia Awards she won the category of Best Actress in a Play for the same work.

 Vilišová joined Prague's National Theatre in 2015.
</item 5>

<item 6>
Maya Deren
Maya Deren (April 29, 1917 – October 13, 1961), born Eleanora Derenkowskaia (Russian: Элеоно́ра Деренко́вская ), was a Russian-American filmmaker and one of the
most important American experimental filmmakers and entrepreneurial promoters of the avant-garde in the 1940s and 1950s.

 Deren was also a choreographer, dancer, film theorist, poet, lecturer, writer and photographer.
</item 6>

<item 7>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 7>

<item 8>
Filmworks X: In the Mirror of Maya Deren
Filmworks X: In the Mirror of Maya Deren features a score for film by John Zorn.

 The album was released on Zorn's own label, Tzadik Records, in 2001 and contains music that Zorn wrote and recorded for the documentary film "In the Mirror of
Maya Deren" on the life and work of Maya Deren directed by Martina Kudlácek.
</item 8>

<item 9>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 9>

<item 10>
Nicholas Mele
Nicholas Mele is an actor who has starred in many movies and on television.

 His first movie role was in the 1976 movie "The Ritz".

 Other movie roles include "Some Kind of Hero" (1982) and "Young Doctors in Love" (1982).

 His most well-known film role was in the 1988 hit horror movie "" as Dennis Johnson.

 A year later, he reprised his role in the hit sequel "".

 His most recent movie is the 2003 film "The Great Gabble".
</item 10>
=== END ITEM id=3eea503fcf524d70 ===

=== ITEM id=2f5be481aa4d485b ===
Gold sentences and facts:
Sentence 0: Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.
  Fact 0.0: Xingcheng is a county-level city of southwest Liaoning province, China.
  Fact 0.1: Xingcheng's former name was Ningyuan (宁远).
  Fact 0.2: Xingcheng has a population of approximately 140,000 urban inhabitants.
  Fact 0.3: Xingcheng is located on Liaodong Bay, the northern coast of the Bohai Sea.
Sentence 1:  It covers an area of 2046 square kilometers and has a population of about one million.
  Fact 1.0: Xingcheng covers an area of 2046 square kilometers.
  Fact 1.1: Xingcheng has a population of about one million.

Retrieved items:
<item 1>
Shanxi–Henan–Shandong Railway
The line can accommodate train speeds of up to 120 km/h .

 Cities along route include Lüliang, Liulin, Hongdong, Changzhi and Pingshun in Shanxi; Linzhou, Anyang, Tangyin, Puyang, Hebi, Fan County and Taiqian County in
Henan; Liangshan County, Dongping, Ningyang, Xintai, Laiwu, Yiyuan, Yishui, Ju County, Junan and Rizhao in Shandong.

 The line was built from 2010 to 2014 and operations began on December 30, 2014.
</item 1>

<item 2>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 2>

<item 3>
Huan River
The Huan River (), or Anyang River (安阳河), is a river in Henan, China, and part of the Hai River basin.

 The river rises north of Linzhou in northwestern Henan, and joins the Wei River near Neihuang in the northeast of the province.
</item 3>

<item 4>
Dafeng, Yancheng, is a coastal Jiangsu district with 112 km of shoreline, a salt-making history, protected ecosystems, and China’s largest Père David’s deer
reserve. Changle, Fuzhou, Fujian, has extensive land and sea areas, Tang-era origins, and became a district in 2017. Fuzhoucheng is a Liaoning town near
highways, a reservoir, and a bird reserve. Jiangyin dialect is a Taihu Wu
</item 4>

<item 5>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 5>

<item 6>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 6>

<item 7>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 7>

<item 8>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 8>

<item 9>
Linzhou, Henan
Linzhou (), formerly Lin County (林县), is a county-level city in Anyang, Henan, China.

 Adjacent to Shanxi Province and Hebei Province, it is located in the north of Henan Province and at the eastern foot of the Taihang Mountains.

 It covers an area of 2046 square kilometers and has a population of about one million.
</item 9>

<item 10>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 10>
=== END ITEM id=2f5be481aa4d485b ===

=== ITEM id=3661f1f2a8543720 ===
Gold sentences and facts:
Sentence 0: Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the United States.
  Fact 0.0: Pago Pago International Airport has IATA code PPG.
  Fact 0.1: Pago Pago International Airport has ICAO code NSTU.
  Fact 0.2: Pago Pago International Airport has FAA LID PPG.
  Fact 0.3: Pago Pago International Airport is also known as Tafuna Airport.
  Fact 0.4: Pago Pago International Airport is a public airport.
  Fact 0.5: Pago Pago International Airport is located 7 miles (11.3 km) southwest of the central business district of Pago Pago.
  Fact 0.6: Pago Pago International Airport is located in the village and plains of Tafuna.
  Fact 0.7: Tafuna is on the island of Tutuila.
  Fact 0.8: Tutuila is in American Samoa.
  Fact 0.9: American Samoa is an unincorporated territory of the United States.
Sentence 1: Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business district of Hoonah, Alaska.
  Fact 1.0: Hoonah Airport has IATA code HNH.
  Fact 1.1: Hoonah Airport has ICAO code PAOH.
  Fact 1.2: Hoonah Airport has FAA LID HNH.
  Fact 1.3: Hoonah Airport is a state-owned public-use airport.
  Fact 1.4: Hoonah Airport is located one nautical mile (2 km) southeast of the central business district of Hoonah.
  Fact 1.5: Hoonah is in Alaska.

Retrieved items:
<item 1>
Tutuila
Tutuila is the largest and the main island of American Samoa in the archipelago of Samoan Islands.

 It is the third largest island in the Samoan Islands chain of the Central Pacific located roughly 4000 km northeast of Brisbane, Australia and over 1200 km
northeast of Fiji.

 It contains a large, natural harbor, Pago Pago Harbor, where Pago Pago, the capital of American Samoa is situated.

 Pago Pago International Airport is also located on Tutuila island.

 Its land expanse is about 68% of the total land area of American Samoa and with 56,000 people accounts for 95% of its population.

 The island has six terrestrial and three marine ecosystems.
</item 1>

<item 2>
Graph query row:
airport: Hoonah Airport
relationship_path: ['LOCATED_IN', 'LOCATED_IN', 'PART_OF']
location: United States
</item 2>

<item 3>
Pago Pago International Airport
Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of
the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the
United States.
</item 3>

<item 4>
Inter Island Airways
Inter Island Airways (also known as "Inter Island Air") is a South Pacific regional airline based in Pago Pago, American Samoa.

 Inter Island Airways operates passenger and cargo flights in and between American Samoa, Independent Samoa and to neighboring Pacific island countries.

 Its main base of operations is at Pago Pago International Airport.
</item 4>

<item 5>
Nu'uuli, American Samoa
Nu'uuli is a village on the central east coast of Tutuila Island, American Samoa.

 It is located on a peninsula several miles up from Pago Pago International Airport.
</item 5>

<item 6>
Tafuna, American Samoa
Tafuna' is a village on the east coast of Tutuila Island, American Samoa.

 It is located on a peninsula a mile north of Pago Pago International Airport and one mile south of Nu'uuli, American Samoa.
</item 6>

<item 7>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 7>

<item 8>
WNG710
WNG710 (sometimes referred to as Pago Pago All Hazards) is a NOAA Weather Radio station that serves the entire United States territory island of American Samoa.

 It is programmed from the National Weather Service Forecast Office in Pago Pago.

 The station broadcasts weather and hazard information for the Eastern, Manu'a, and Western Districts, which are the three political divisions of American
Samoa.

 WNG710 has a sister station, WZ2529 at Mt. Olotele.

 Its transmitter is located at the opposite side of the island.
</item 8>

<item 9>
Veterans Memorial Stadium (Pago Pago)
Veterans Memorial Stadium is a football stadium located in Pago Pago Park, in Pago Pago, American Samoa.

 The 10,000 capacity venue is one of the smallest stadiums in Oceania, and serves as American Samoa's national stadium.

 It is the home venue of the American Samoa national football team, hosting all of their home games.

 It is currently used mostly for matches in various football codes, such as soccer, rugby league, and the territory's most popular code, American football.
</item 9>

<item 10>
Pago Pago
Pago Pago ( ; Samoan: ] ) is the territorial capital of American Samoa.

 It is on the main island of American Samoa, Tutuila.

 The territory is served by Pago Pago International Airport at Tafuna, some 8 miles south west of Pago Pago.

 Tourism, entertainment, food, and tuna canning are its main industries.
</item 10>
=== END ITEM id=3661f1f2a8543720 ===

=== ITEM id=5c528577fcbe15eb ===
Gold sentences and facts:
Sentence 0: Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.
  Fact 0.0: Ted Kooshian was born October 8, 1961.
  Fact 0.1: Ted Kooshian is an American jazz pianist.
  Fact 0.2: Ted Kooshian is an American jazz keyboardist.
  Fact 0.3: Ted Kooshian is an American jazz composer.
  Fact 0.4: Ted Kooshian has performed with Aretha Franklin.
  Fact 0.5: Ted Kooshian has performed with Chuck Berry.
  Fact 0.6: Ted Kooshian has performed with Marvin Hamlisch.
  Fact 0.7: Ted Kooshian has performed with Sarah Brightman.
  Fact 0.8: Ted Kooshian has performed with Il Divo.
  Fact 0.9: Ted Kooshian has performed with Blood, Sweat, and Tears.
Sentence 1:  Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
  Fact 1.0: Marvin Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
Sentence 2:  This collection of all four is referred to as an "EGOT".
  Fact 2.0: The collection of all four awards (Emmy, Grammy, Oscar and Tony) is referred to as an 'EGOT'.

Retrieved items:
<item 1>
Grammy Award for Best Male Rap Solo Performance
The Grammy Award for Best Male Rap Solo Performance was an honor presented to male recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958, and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 1>

<item 2>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 2>

<item 3>
Lonesome Pine Fiddlers
This most talented family of musicians were the best, surpassed by none.

 Finally, on October 1, 2009, The Lonesome Pine Fiddlers got their due when they were inducted into the International Bluegrass Hall of Fame at the Ryman
Theater (the original home of the Grand Ole Opry).

 Bobby Osborne, Melvin Goins and Paul Williams were there to receive the bands award.

 In the crowd of a sold out theater was the son of Ezra Cline, Scotty Ireland Cline, who recalled being in that same theater as a child sitting on stage and
watching the Fiddlers play.

 (At the time, the Opry had bleachers for family just off stage).

 The final act of the evening at the IBMA Awards was the playing of "Pain in my Heart" by Osborne, Goins and Williams along with a Song from the Dillards, who
were also inducted the same evening.
</item 3>

<item 4>
Marvin Hamlisch
Marvin Frederick Hamlisch (June 2, 1944August 6, 2012) was an American composer and conductor.

 Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.

 This collection of all four is referred to as an "EGOT".

 He is one of only two people (along with Richard Rodgers) to have won those four prizes and a Pulitzer Prize.
</item 4>

<item 5>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 5>

<item 6>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 6>

<item 7>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 7>

<item 8>
List of awards and nominations received by Kim Tae-yeon
List of awards and nominations received by Kim Tae-yeon
</item 8>

<item 9>
Academy Honorary Award
The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards (previously called the Special Award, which was first presented in early 1929) – is
given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not
covered by existing Academy Awards, although prior winners of competitive Academy Awards are not excluded from receiving the Honorary Award.

 Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.

 Unlike the Special Achievement Award instituted in 1972 (and discontinued in 1995), those on whom the Academy confers its Honorary Award do not have to meet
"the Academy's eligibility year and deadline requirements."

 Like the Special Achievement Award, the Special Award and Honorary Award have been used to reward significant achievements of the year that did not fit in
existing categories, subsequently leading the Academy to establish several new categories, and to honor exceptional career achievements, contributions to the
motion picture industry, and service to the Academy.
</item 9>

<item 10>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 10>
=== END ITEM id=5c528577fcbe15eb ===

=== ITEM id=dc33422c50a3c0f1 ===
Gold sentences and facts:
Sentence 0: Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.
  Fact 0.0: Henderson Point is a census-designated place (CDP).
  Fact 0.1: Henderson Point is in Harrison County, Mississippi, United States.
Sentence 1:  Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
  Fact 1.0: Gulfport is a city adjoining Biloxi.
  Fact 1.1: Biloxi is a county seat of Harrison County.

Retrieved items:
<item 1>
Hendersonville, North Carolina
Hendersonville is a city in Henderson County, North Carolina United States.

 This city is 22 miles south of Asheville.

 It is the county seat of Henderson County.

 Like the county, the city is named for 19th-century North Carolina Supreme Court Chief Justice Leonard Henderson.
</item 1>

<item 2>
Biloxi, Mississippi
Biloxi ( ) is a city in Harrison County, Mississippi, United States.

 The 2010 United States Census recorded the population as 44,054, and in 2016 the estimated population was 45,975.

 Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
</item 2>

<item 3>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 3>

<item 4>
Gwendolyn Elaine Armstrong
Gwendolyn Elaine Armstrong was a black Mississippi pioneer in the Civil Rights Movement.

 In September, 1965, she and Raylawni Branch, both local natives, integrated the University of Southern Mississippi at Hattiesburg.

 They thus completed the process of breaking the segregation barriers at Mississippi’s universities which had been begun by Clyde Kennard at (then) Mississippi
Southern College (1956–61) and carried forward by James Meredith at the University of Mississippi (September, 1962) and Richard Holmes at Mississippi State
University (July, 1965).
</item 4>

<item 5>
Henderson, Texas
Henderson is a city in Rusk County, northeast Texas, United States.

 The population was 13,712 at the 2010 census.

 It is the county seat of Rusk County.

 Henderson is named for James Pinckney Henderson, the first governor of Texas.
</item 5>

<item 6>
Walpole (CDP), Massachusetts
Walpole is a small census-designated place (CDP) located within the much larger town (21 square miles in size) of Walpole in Norfolk County, Massachusetts,
United States.

 Walpole Town as it is called by the US Census Bureau, also includes a much larger population (24,070 in 2010).

 The population of the Census Designated Place was 5,918 at the 2010 census.
</item 6>

<item 7>
Henderson Point, Mississippi
Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.

 It is part of the Gulfport–Biloxi Metropolitan Statistical Area.

 The population was 170 at the 2010 census.
</item 7>

<item 8>
Paulding, Mississippi
Paulding is an unincorporated community in and one of the two county seats of Jasper County, Mississippi.

 It is the only unincorporated county seat in Mississippi.

 Settled in 1833, it was named by United States settlers in honor of Revolutionary War hero John Paulding.

 After its citizens refused to contribute to a new railroad, the community was bypassed in favor of Bay Springs, Mississippi, which was designated a railroad
stop to the west and the second county seat.

 It attracted more development and industry.
</item 8>

<item 9>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 9>

<item 10>
Piney Woods Country Life School
The Piney Woods Country Life School (or The Piney Woods School) is a co-educational independent historically African-American boarding school for grades 9-12 in
Piney Woods, unincorporated Rankin County, Mississippi.

 It is 21 mi south of Jackson.

 It is one of four remaining historically African-American boarding schools in the United States.

 It is currently the largest African-American boarding school, as well as being the second oldest continually operating African-American boarding school.
</item 10>
=== END ITEM id=dc33422c50a3c0f1 ===
