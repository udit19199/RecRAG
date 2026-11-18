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

=== ITEM id=6bc072f1b47f0c7a ===
Gold sentences and facts:
Sentence 0: Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.
  Fact 0.0: Giovanni 'Tinto' Brass was born on 26 March 1933.
  Fact 0.1: Giovanni 'Tinto' Brass is an Italian filmmaker.
Sentence 1: Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.
  Fact 1.0: Baltasar Kormakur Samper was born on 27 February 1966.
  Fact 1.1: Baltasar Kormakur Samper is an Icelandic actor.
  Fact 1.2: Baltasar Kormakur Samper is an Icelandic theater director.
  Fact 1.3: Baltasar Kormakur Samper is an Icelandic film director.
  Fact 1.4: Baltasar Kormakur Samper is an Icelandic film producer.

Retrieved items:
<item 1>
Actor John Steiner became a fixture in Italian genre films, including works by Tinto Brass. Ruggero Deodato, Italian director, writer and actor, reportedly
directed most of 1964’s horror-inflected peplum *Hercules, Prisoner of Evil* while credited assistant director Antonio Margheriti was occupied elsewhere.
Deodato’s *Cannibal Holocaust* (1980), filmed in Colombia, drew on staged-seeming news coverage and helped drive the 197
</item 1>

<item 2>
Baltasar Breki Samper
Baltasar Breki Samper (born 22 July 1989), sometimes referred to as Baltasar Breki Baltasarsson, is an Icelandic actor.

 He is the son of Icelandic director Baltasar Kormákur.

 Before graduating from the Icelandic Academy of the Arts in 2015 Samper helped his father on several films as second assistant director.

 He is known for his role as Hjörtur in the Icelandic TV series "Trapped".
</item 2>

<item 3>
Lev Kuleshov (1899–1970), Soviet filmmaker and theorist, pioneered Soviet montage, the Kuleshov effect, and creative geography. Hermann Brunn (1862–1939),
German mathematician, contributed to convex geometry and knot theory. Mercury’s 165-km Chaikovskij crater honors composer Pyotr Tchaikovsky; a 1970 Soviet
biopic about him earned two Oscar nominations. Søren Kierkegaard (
</item 3>

<item 4>
Fermo posta Tinto Brass
Fermo posta Tinto Brass is a 1995 Italian comedy film-erotic film directed by Tinto Brass and set in vignettes.
</item 4>

<item 5>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 5>

<item 6>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 6>

<item 7>
Baltasar Kormákur
Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.

 He is best known for directing the films "101 Reykjavík", "Hafið", "A Little Trip to Heaven" (starring Julia Stiles and Forest Whitaker), a film based on the
book "Mýrin" "(Jar City)" by Arnaldur Indriðason, "Contraband", "2 Guns" (starring Mark Wahlberg and Denzel Washington) and "Everest".

 His father is the Spanish painter Baltasar Samper and actor Baltasar Breki Samper is his son.
</item 7>

<item 8>
Austrian director-screenwriter Stefan Ruzowitzky made *The Inheritors* (1998), Austria’s unsuccessful Oscar submission, and *The Counterfeiters* (2007), a WWII
drama about Nazi forgery at Sachsenhausen that won the foreign-language-film Oscar; Devid Striesow played Bernhard Krüger. German director Wolfgang Becker is
known for *Good Bye, Lenin!* (2003), starring Daniel Brühl and Katrin Saß. Other entries
</item 8>

<item 9>
The entries survey European films and music drama, 1913–1961. Swedish director Mauritz Stiller’s works include *The Blizzard* (a student saves his family
estate), *Erotikon* (a professor’s wife attracts suitors), and *The Saga of Gösta Berling*, Greta Garbo’s breakout film. Hungarian director Béla Gaál made
comedies including *The Dream Car*. Other highlights: *Triumph of the Will*, Riefenstahl’s
</item 9>

<item 10>
Contraband (2012 film)
Contraband is a 2012 American crime thriller film directed by Baltasar Kormákur, starring Mark Wahlberg, Kate Beckinsale, Ben Foster, Caleb Landry Jones,
Giovanni Ribisi, Lukas Haas, Diego Luna and J. K. Simmons.

 The film is a remake of the 2008 Icelandic film "Reykjavík-Rotterdam" which Baltasar Kormákur starred in.

 It was released on January 13, 2012 in the United States by Universal Pictures.
</item 10>
=== END ITEM id=6bc072f1b47f0c7a ===

=== ITEM id=ea2579ed58f13fc4 ===
Gold sentences and facts:
Sentence 0: Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming pieces and a game system.
  Fact 0.0: Icehouse pieces are also called Icehouse Pyramids.
  Fact 0.1: Treehouse pieces are also called Treehouse Pyramids.
  Fact 0.2: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, and Treehouse Pyramids are officially called Looney Pyramids.
  Fact 0.3: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are nestable pyramid-shaped gaming pieces.
  Fact 0.4: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are stackable pyramid-shaped gaming pieces.
  Fact 0.5: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are a game system.
Sentence 1:  The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
  Fact 1.0: The Looney Pyramids game system was invented by Andrew Looney.
  Fact 1.1: The Looney Pyramids game system was invented by John Cooper.
  Fact 1.2: The Looney Pyramids game system was invented in 1987.
  Fact 1.3: The Looney Pyramids game system was originally for use in the game of Icehouse.
Sentence 2:  In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
  Fact 2.0: "Kill Doctor Lucky" won the Origins Award in 1998.
  Fact 2.1: "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".

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
Treehouse (game)
Treehouse is a game in which players try to get their configuration of Icehouse pieces to match the central configuration, shared by all players.

 The rolling of the special "Treehouse Die" tells the player what kind of move to make to change his own or the central configuration, and then he does so to
best move towards the goal.
</item 2>

<item 3>
Icehouse (song)
"Icehouse" is a song by the Australian rock band Flowers.

 It was released as a single in Europe in 1982 by Chrysalis Records from the band's first album, "Icehouse", after the band changed its name to Icehouse.

 In the United States, the song peaked at number 28 on the "Billboard" Top Tracks chart in 1981.
</item 3>

<item 4>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 4>

<item 5>
Icehouse pieces
Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming
pieces and a game system.

 The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
</item 5>

<item 6>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 6>

<item 7>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 7>

<item 8>
Boxes (Icehouse album)
Boxes is a soundtrack album by Australian band Icehouse, released by Festival Records / Chrysalis Records in November 1985.

 The work was originally conceived by its composers, Iva Davies and Robert Kretschmer, in collaboration with choreographer Graeme Murphy of the Sydney Dance
Company for performance as the ballet "Boxes".

 The first live performance of "Boxes" was given by the Sydney Dance Company together with Davies and Kretschmer of Icehouse and guest percussionist Masaki
Tanazawa in the Opera Theatre of the Sydney Opera House on 7 November 1985.

 "Boxes" was released in the US under the name Sydney Dance Company.

 The single "No Promises" was released as a 12" in November and a cassingle in December on Regular Records for Australian and New Zealand markets, it contained
other tracks from "Boxes"; two tracks from "Boxes" were later released on their next studio album "Measure for Measure" in April 1986.
</item 8>

<item 9>
Hidden Expedition
Hidden Expedition is a series of single player hidden object casual games developed by the internal studios of Big Fish Games for the first five installments
(using Big Fish Games Framework as the engine, and with the help of Flood Light Games in the 5th game), and by Eipix Entertainment for all subsequent
installments.

 As of September 2017, a total of fifteen games in the series have been released.

 The Hidden Expedition series marks the second major hidden object game brand from Big Fish Games.

 The second game in the series, "Hidden Expedition: Everest", would go on to be the first game Big Fish released to the iPhone.
</item 9>

<item 10>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 10>
=== END ITEM id=ea2579ed58f13fc4 ===

=== ITEM id=3fb5735c13e1b7fd ===
Gold sentences and facts:
Sentence 0:  The film stars Juno Temple, Julia Garner, Alessandro Nivola, Maggie Siff, Olivia Luccardi and Philip Ettinger.
  Fact 0.0: The film stars Juno Temple.
  Fact 0.1: The film stars Julia Garner.
  Fact 0.2: The film stars Alessandro Nivola.
  Fact 0.3: The film stars Maggie Siff.
  Fact 0.4: The film stars Olivia Luccardi.
  Fact 0.5: The film stars Philip Ettinger.
Sentence 1:  She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight Rises".
  Fact 1.0: The actress has appeared in the film Black Mass.
  Fact 1.1: The actress has appeared in the film The Other Boleyn Girl.
  Fact 1.2: The actress has appeared in the film Wild Child.
  Fact 1.3: The actress has appeared in the film Atonement.
  Fact 1.4: The actress has appeared in the film Maleficent.
  Fact 1.5: The actress has appeared in the film The Three Musketeers.
  Fact 1.6: The actress has appeared in the film The Dark Knight Rises.

Retrieved items:
<item 1>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 1>

<item 2>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 2>

<item 3>
Dirty Girl (2010 film)
Dirty Girl is a 2010 American coming of age comedy-drama film written and directed by Abe Sylvia.

 It stars Juno Temple, Milla Jovovich and William H. Macy.

 It premiered at the 2010 Toronto International Film Festival on September 12, 2010.

 It was distributed theatrically by The Weinstein Company on October 7, 2011.
</item 3>

<item 4>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 4>

<item 5>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
</item 5>

<item 6>
Kate Winslet, a British actress and singer, debuted as Juliet Hulme in *Heavenly Creatures* (1994). Her early films included *Sense and Sensibility* (1995),
earning a Best Supporting Actress Oscar nomination; *Jude* and *Hamlet* (1996); and *Titanic* (1997), which brought Best Actress nominations from the Academy,
Golden Globe, and Screen Actors Guild Awards.
</item 6>

<item 7>
Juliet Weber works in the film industry as a producer, director, writer, and editor. She co-produced *Fastpitch* (2000) and *Unchained Memories* (2003).
</item 7>

<item 8>
The entries profile performers and productions, including *Doctor Who*, *EastEnders* and *Waterloo Road*. Juliet Weber is a producer, director, writer and
editor. Juno Temple starred in *Vinyl*; Holly Hunter won major awards for *The Piano*. Kathryn Drysdale played Louise Brooks and Meghan Markle. Kate Winslet
rose to prominence with *Heavenly Creatures* and *Titanic*. Cate Blanchett won two Oscars, including for *The Aviator*. Reese
</item 8>

<item 9>
Cracks (film)
Cracks is a 2009 independent drama psychological thriller film starring Eva Green, James McGovern, Juno Temple, María Valverde, and Imogen Poots, which was
released theatrically in the UK and Ireland on 4 December 2009.

 In the United States it was released by IFC Films on 18 March 2011, and premiered on Showtime in late 2011.
</item 9>

<item 10>
Juno Temple (born 21 July 1989) is an English actress known for films including *Black Mass*, *The Other Boleyn Girl*, *Wild Child*, *Atonement*, *Maleficent*,
*The Three Musketeers* and *The Dark Knight Rises*. She starred in HBO’s period drama *Vinyl*.
</item 10>
=== END ITEM id=3fb5735c13e1b7fd ===

=== ITEM id=6a731ac96c1d5fd1 ===
Gold sentences and facts:
Sentence 0: Fastbacks were a Seattle, Washington, punk rock band.
  Fact 0.0: Fastbacks were a punk rock band.
  Fact 0.1: Fastbacks were from Seattle, Washington.
Sentence 1: Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.
  Fact 1.0: Local Natives is an American indie rock band.
  Fact 1.1: Local Natives was previously known as Cavil at Rest.
  Fact 1.2: Local Natives is based in Silver Lake, Los Angeles, California, United States.

Retrieved items:
<item 1>
Fastbacks
Fastbacks were a Seattle, Washington, punk rock band.

 Formed in 1979 by songwriter/guitarist Kurt Bloch (born August 28, 1960), and friends Lulu Gargiulo (guitar and vocals, born October 12, 1960) and Kim Warnick
(bass and vocals, born April 7, 1959), they disbanded in 2001.

 Their sound mixed a generally punk rock approach to vocals and sound textures with poppy tunes and strong musicianship.
</item 1>

<item 2>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 2>

<item 3>
Hummingbird (Local Natives album)
Hummingbird is the second studio album by American indie rock band Local Natives, released on January 29, 2013 on Frenchkiss Records in the US, and on
Infectious Records in Europe.

 Produced by The National guitarist Aaron Dessner, the album was preceded by the singles, "Breakers" and "Heavy Feet."

 "Breakers" reached #12 on the Billboard Top 200.

 The song "Mt. Washington" was also featured in the 2015 video game "Life Is Strange".
</item 3>

<item 4>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 4>

<item 5>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 5>

<item 6>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 6>

<item 7>
Chief (band)
Chief is a four-person band from Santa Monica, California, stationed in Los Angeles, California.

 Chief has released three singles entitled "Mighty Proud," "Breaking Walls", and "Night And Day," as well as an EP called "The Castle Is Gone" and their debut
and only full album, "Modern Rituals".

 All of these releases (except for the EP which was released under their own record label) have been via Domino Records.

 They have been compared to bands such as Coldplay and Local Natives by "The New York Times".

 The "Times" goes on to say "Mr. Koga can sound like the young Tom Petty."

 Other reviews of Chief have similarly been generally good.

 On June 14, 2011, they played their farewell show at The Troubadour but began performing together again one year later.
</item 7>

<item 8>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 8>

<item 9>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 9>

<item 10>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 10>
=== END ITEM id=6a731ac96c1d5fd1 ===

=== ITEM id=6b0e1c4c1abc5534 ===
Gold sentences and facts:
Sentence 0:  He creates bespoke millinery under the J Smith Esquire brand for a portfolio of private clients that includes Angelina Jolie, Brad Pitt and Amal Clooney.
  Fact 0.0: This person creates bespoke millinery under the J Smith Esquire brand.
  Fact 0.1: This person creates bespoke millinery for a portfolio of private clients.
  Fact 0.2: The portfolio of private clients includes Angelina Jolie.
  Fact 0.3: The portfolio of private clients includes Brad Pitt.
  Fact 0.4: The portfolio of private clients includes Amal Clooney.
Sentence 1: Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and human rights.
  Fact 1.0: Amal Clooney is also known by the name Alamuddin (née Alamuddin).
  Fact 1.1: Amal Clooney's Arabic name is أمل علم الدين.
  Fact 1.2: Amal Clooney was born on 3 February 1978.
  Fact 1.3: Amal Clooney is a barrister at Doughty Street Chambers.
  Fact 1.4: Amal Clooney specialises in international law.
  Fact 1.5: Amal Clooney specialises in human rights.

Retrieved items:
<item 1>
Doughty Street Chambers
Doughty Street Chambers is a British set of barristers' chambers situated in Bristol, Manchester and London's Doughty Street, undertaking criminal justice,
public law, immigration, employment, human rights and civil liberties work.
</item 1>

<item 2>
Amal Clooney
Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and
human rights.

 Her clients include Julian Assange, the founder of WikiLeaks, in his fight against extradition.

 She has also represented the former prime minister of Ukraine, Yulia Tymoshenko, and Egyptian-Canadian journalist Mohamed Fahmy.

 She is married to the American actor George Clooney.
</item 2>

<item 3>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 3>

<item 4>
Barbados Bar Association
The Barbados Bar Association is a voluntary association of attorneys in Barbados who practise at the independent bar as barristers and Queen's Counsel.

 It was created by the Barbados Bar Association Act of 1940.
</item 4>

<item 5>
Mónica Feria Tinta
Mónica Feria Tinta is a leading public international lawyer.

 She practises as a Barrister at 20 Essex Street Chambers.

 In 2000 she became the first Peruvian-born lawyer to receive the Diploma of the Hague Academy of International Law in history.

 Her litigation work led to the first international human rights court decision ordering the prosecution of a former Head of State for crimes under
international law.

 She was awarded the Inge Genefke International Award for her work as an international lawyer in 2006 and was co-recipient of the Gruber Justice Prize 2007, for
her contributions advancing the cause of justice as delivered through the legal system.

 She is a member of the Bar of England and Wales, and the American Society of International Law.
</item 5>

<item 6>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 6>

<item 7>
Edmund Bartley-Denniss
Sir Edmund Robert Bartley Bartley-Denniss KC (9 April 1854 – 20 March 1931), born Edmund Robert Bartley Denniss, was a barrister, prominent Freemason and
Conservative Party Member of Parliament in the United Kingdom.

 He was also a pioneer of the sport of cycling in Britain.
</item 7>

<item 8>
Pupil master
Occasionally, a pupil barrister may undertake a third "six", extending the training period a further six months.

 At the end of pupillage, to continue practising the law, a barrister may attempt to become a tenant in a set of barristers' chambers, or find a position as an
employed barrister.
</item 8>

<item 9>
Pupil master
A pupil master ('pupilmaster' or 'pupil-master') or in the case of a female barrister, 'pupil mistress' etc., is the former name given to an experienced
barrister who a pupil shadows during their pupillage.

 The term pupil master or mistress has now been replaced by the term 'pupil supervisor'.

 Barristers are called to the Bar via one of the four Inns of Court upon successful completion of the BPTC and having undertaken a required number of
"qualifying sessions" in their chosen Inn of Court.

 In most cases, the newly called barrister is then required to undertake training for a period of at least a year before the barrister can start their own
private practice.

 This training period is known as pupillage, usually split into two periods of six months known as "sixes".

 The first "six" is a non-practising six, during which the pupil will shadow their pupil master; the second is usually a practising "six", when the pupil, with
their pupil master's permission, can undertake the supply of legal services and exercise rights of audience in court.
</item 9>

<item 10>
Shahram Taghavi
Shahram Taghavi is a barrister practising in the United Kingdom.

 He is a specialist in Judicial Review, Human Rights, Immigration and EU law.

 He was called to the England & Wales bar in 1994.

 He is a partner at Article 1, a law firm specialising in immigration, EU freedoms of movement and human rights law.

 He practised as an independent barrister at Doughty Street Chambers specialising in immigration and public law.

 He was head of both the Immigration and Public Law departments at Simons Muirhead & Burton solicitors.

 He was a Senior Barrister at Bates Wells & Braithwaite LLP as a member of their Public & Regulatory and Immigration departments, Deputy Head of Immigration at
Lewis Silkin LLP, Head of Immigration at Charles Russell LLP (now Charles Russell Speechly LLP) and National Head of Immigration at Grant Thornton LLP.
</item 10>
=== END ITEM id=6b0e1c4c1abc5534 ===

=== ITEM id=3ea941320535aaa6 ===
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
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 1>

<item 2>
Charles Andrew Whitney
Charles Andrew Whitney (November 14, 1864- December 31, 1912) was an American businessman and industrialist in the late 19th century, born in Princeton,
Massachusetts.

 He was part of the prominent American Whitney family.

 In 1859, Charles, his brother Levi L. Whitney, and Orville E. Thompson helped lead the large-scale manufacturing of leather boots and shoes in Chicago and were
attributed with successfully running the first factory of its kind there.

 Whitney himself held patents for the manufacture of leather.
</item 2>

<item 3>
The Andy Warhol Story
The Andy Warhol Story (1966) is an underground film directed by Andy Warhol with cinematography by Paul Morrissey, and starring Edie Sedgwick and Rene Ricard
(as Andy Warhol).
</item 3>

<item 4>
John Warhola
John Warhola (May 31, 1925 – December 24, 2010) played a pivotal role in maintaining the legacy of his younger brother, pop artist Andy Warhol, assigned
responsibility by their father on his deathbed to ensure that Andy attended college and serving as a trustee of the Andy Warhol Foundation for the Visual Arts
after his brother's death in 1987.

 Warhola oversaw the establishment of The Andy Warhol Museum in Pittsburgh and the Andy Warhol Museum of Modern Art in Medzilaborce, Slovakia.
</item 4>

<item 5>
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
</item 5>

<item 6>
Green Coca-Cola Bottles
With respect to its artistic merits, Warhol’s painting utilizes repetitive imagery in the shape of hundred and twelve almost identical bottles, evoking a sense
of mechanicalization which overlaps his former usage of mass culture objects.

 The widespread display of advertisement flooded the American public during the sixties, and Warhol had been successful in mapping it onto his canvas, no wonder
it is considered one of Warhol’s masterpieces.
</item 6>

<item 7>
Andy Warhol
Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual
art movement known as pop art.

 His works explore the relationship between artistic expression, celebrity culture, and advertising that flourished by the 1960s, and span a variety of media,
including painting, silkscreening, photography, film, and sculpture.

 Some of his best known works include the silkscreen paintings "Campbell's Soup Cans" (1962) and "Marilyn Diptych" (1962), the experimental film "Chelsea Girls"
(1966), and the multimedia events known as the "Exploding Plastic Inevitable" (1966–67).
</item 7>

<item 8>
Beautiful Darling
Beautiful Darling: The Life and Times of Candy Darling, Andy Warhol Superstar is a 2010 feature-length documentary film about Candy Darling, the transgender
pioneer, actress and Andy Warhol Superstar.

 The film was written and directed by James Rasin and features Chloë Sevigny as "the voice of Candy Darling", reading from Candy's private diaries and letters.

 Patton Oswalt voices Andy Warhol and Truman Capote.

 Louis Durra composed the score.
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
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 10>
=== END ITEM id=3ea941320535aaa6 ===

=== ITEM id=e2c796efb2024255 ===
Gold sentences and facts:
Sentence 0:  They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.
  Fact 0.0: This team is most well known for including Sugar Rashad Evans.
  Fact 0.1: Sugar Rashad Evans is a former UFC light heavyweight champion.
  Fact 0.2: This team is most well known for including Vitor Belfort.
  Fact 0.3: Vitor Belfort is a former UFC light heavyweight champion.
  Fact 0.4: This team is most well known for including Eddie Alvarez.
  Fact 0.5: Eddie Alvarez is a former UFC lightweight champion.
Sentence 1: Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.
  Fact 1.0: Vitor Vieira Belfort was born on April 1, 1977.
  Fact 1.1: Vitor Vieira Belfort is a Brazilian mixed martial artist.
  Fact 1.2: Vitor Vieira Belfort is currently competing in the Middleweight division of the Ultimate Fighting Championship.

Retrieved items:
<item 1>
Vitor Belfort
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 He is the former UFC Light Heavyweight Champion, UFC 12 Heavyweight Tournament Champion, and the Cage Rage World Light Heavyweight Champion.

 As of June 16, 2017, he is the #11 contender in official UFC Middleweight rankings.
</item 1>

<item 2>
Jack Marshman
Jack "Hammer" Marshman (born December 19, 1989) is a Welsh mixed martial artist.

 He was the Middleweight Cage Warriors Fighting Championship (CWFC) champion and Lonsdale Middleweight Champion.

 He currently competes in Middleweight division in the Ultimate Fighting Championship(UFC).
</item 2>

<item 3>
Sam Alvey
Samuel Alvey (born May 6, 1986) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.

 A professional MMA competitor since 2008, Alvey has also formerly competed for Bellator, King of the Cage, the MFC, and was a contestant on .

 As of March 2017, Alvey is #14 in the official UFC Middleweight Rankings.
</item 3>

<item 4>
Paulo Borrachinha
Paulo Henrique Costa (born April 21, 1991), better known as Paulo Borrachinha, is a Brazilian mixed martial artist currently competing in the Middleweight
division of the Ultimate Fighting Championship.

 In the day August 3 of 2017, Borracinha and the Clube Atlético Mineiro officialized institutional partnership.

 The agreement did not create a bond between the club and the athlete.

 The agreement allows the use of the athletic shield of Atlético in the training uniforms of the fighter, within the possibilities of his contract with the UFC.
</item 4>

<item 5>
Luke Zachrich
Luke Zachrich (born October 1, 1981) is an American mixed martial artist and former professional boxer who formerly competed in the Middleweight division of the
Ultimate Fighting Championship A professional mixed martial arts competitor since 2006, Zachrich was a member of Team Forrest on the seventh season of The
Ultimate Fighter, and has also fought for King of the Cage, Xtreme Fighting Organization, Bellator, and Ultimate Victory Challenge, the latter of which he was
the inaugural Middleweight Champion.
</item 5>

<item 6>
David Branch (fighter)
David Branch (born September 26, 1981) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2007, he has also formerly competed for the World Series of Fighting, Bellator, Shark Fights, and Titan FC.

 Branch was the inaugural World Series of Fighting Light Heavyweight and Middleweight Champion.

 He is currently ranked as the #9 Middleweight in the world according to Sherdog.
</item 6>

<item 7>
Blackzilians
The Blackzilians is a defunct professional team of fighters in mixed martial arts, boxing, kickboxing, amateur wrestling and catch wrestling located in South
Florida.

 They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC
lightweight champion Eddie Alvarez.
</item 7>

<item 8>
Kelvin Gastelum
Kelvin Gastelum (born October 24, 1991) is an American professional mixed martial artist who is currently signed to the Ultimate Fighting Championship (UFC).

 He was the winner of "The Ultimate Fighter 17".

 Gastelum is currently competing as a middleweight and is ranked #8 in the UFC official middleweight rankings.
</item 8>

<item 9>
Rashad Evans
Rashad Anton Evans (born September 25, 1979) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2003, Evans was the Heavyweight winner of The Ultimate Fighter 2 and is also a former UFC Light Heavyweight Champion.
</item 9>

<item 10>
Luke Barnatt
Luke Barnatt (born 13 April 1988) is an English mixed martial artist who fought as a Middleweight for the Ultimate Fighting Championship.

 He was a member of FX's , and has also competed for BAMMA and UCMMA in England.

 At 6' 6" Barnatt is one of the tallest fighters currently competing in his weight class.

 Barnatt received his nickname "Bigslow" from brothers John Maguire and Tommy Maguire, for his size and being slow in response to their jokes and insults.

 Barnatt competes in the middleweight division of the Absolute Championship Berkut.
</item 10>
=== END ITEM id=e2c796efb2024255 ===

=== ITEM id=1939d16678154d7a ===
Gold sentences and facts:
Sentence 0: Ionolyce is a small genus of butterflies in the family Lycaenidae.
  Fact 0.0: Ionolyce is a small genus of butterflies.
  Fact 0.1: Ionolyce is in the family Lycaenidae.
Sentence 1:  "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.
  Fact 1.0: I. helicon is widespread in the Indoaustralasian region.
  Fact 1.1: I. helicon is widespread from India to the Malay Archipelago.
Sentence 2: India lies on the Indian Plate, the northern portion of the Indo-Australian Plate, whose continental crust forms the Indian subcontinent.
  Fact 2.0: India lies on the Indian Plate.
  Fact 2.1: The Indian Plate is the northern portion of the Indo-Australian Plate.
  Fact 2.2: The continental crust of the Indian Plate forms the Indian subcontinent.
Sentence 3:  It is the seventh-largest country in the world, with a total area of 3287263 km2 .
  Fact 3.0: India is the seventh-largest country in the world.
  Fact 3.1: India has a total area of 3287263 km2.

Retrieved items:
<item 1>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 1>

<item 2>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 2>

<item 3>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 3>

<item 4>
Honeyguide
Honeyguides (family Indicatoridae) are near passerine bird species of the order Piciformes.

 They are also known as indicator birds, or honey birds, although the latter term is also used more narrowly to refer to species of the genus "Prodotiscus".

 They have an Old World tropical distribution, with the greatest number of species in Africa and two in Asia.

 These birds are best known for their interaction with humans.

 Honeyguides are noted and named for one or two species that will deliberately lead humans (but, contrary to popular claims, not honey badgers) directly to bee
colonies, so that they can feast on the grubs and beeswax that are left behind.
</item 4>

<item 5>
Indian muntjac
The Indian muntjac ("Muntiacus muntjak"), also called red muntjac and barking deer, is a common muntjac deer species in South and Southeast Asia.

 It is listed as Least Concern on the IUCN Red List.
</item 5>

<item 6>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 6>

<item 7>
Dregea volubilis
Dregea volubilis is a species of plant in the Apocynaceae family that is native to India.
</item 7>

<item 8>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 8>

<item 9>
Ionolyce
Ionolyce is a small genus of butterflies in the family Lycaenidae.

 "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.

 "I selkon"and "I.brunnescens" are both endemic to the Solomon Islands.
</item 9>

<item 10>
Consul fabius
Consul fabius, the tiger leafwing, is the most common and well known species of the genus "Consul" of subfamily Charaxinae in the brush-footed butterfly family
(Nymphalidae).

 It is found all over the Neotropics.
</item 10>
=== END ITEM id=1939d16678154d7a ===

=== ITEM id=35b98951c0989064 ===
Gold sentences and facts:
Sentence 0: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 0.0: California State Polytechnic University, Pomona is also called CPP, Cal Poly Pomona, or Cal Poly.
  Fact 0.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 0.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.
Sentence 1:  It is one of two polytechnics in the California State University system.
  Fact 1.0: California State Polytechnic University, Pomona is one of two polytechnics in the California State University system.
Sentence 2:  Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.
  Fact 2.0: The university is named after the first President of Russia, B.N. Yeltsin.
  Fact 2.1: The full name of the university is 'Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina'.
  Fact 2.2: The university, Ural Federal University, is often shortened to UrFU or УрФУ.
  Fact 2.3: Ural Federal University was formed by a merger of the Ural State Technical University and Ural State University.
  Fact 2.4: Ural Federal University is one of the leading educational institutions in the Ural region.
Sentence 3:  Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.
  Fact 3.0: Ural Federal University acts as a research and innovation center of the Ural region.
  Fact 3.1: Ural Federal University has close cooperation with the Russian Academy of Sciences.

Retrieved items:
<item 1>
Ural Federal University
The links between the university and intermediate educational institutions are actively developed.

 The University also performs consistent work on establishing strategic partnership with employers for the sake of development of the regional economy:
employers are involved in the educational environment as direct participants, large-scale enployers-sponsored and corporate education is carried out based on
the client-centered approach, joint business projects in economic and social areas are realized.
</item 1>

<item 2>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 2>

<item 3>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 3>

<item 4>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 4>

<item 5>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 5>

<item 6>
Ural Federal University
The Ural Federal University named after the first President of Russia B. N. Yeltsin (Russian: Уральский федеральный университет имени первого Президента России
Б.Н.

 Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural
State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.

 Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.

 Training of students is carried out in four main areas of knowledge and 108 academic majors.

 The links between the university and intermediate educational institutions are actively developed.
</item 6>

<item 7>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 7>

<item 8>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 8>

<item 9>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 9>

<item 10>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 10>
=== END ITEM id=35b98951c0989064 ===

=== ITEM id=df1475aeabb41581 ===
Gold sentences and facts:
Sentence 0: Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister, and for two terms President of the General Assembly of the League of Nations (1930–32).
  Fact 0.0: Nicolae Titulescu was born on March 4, 1882.
  Fact 0.1: Nicolae Titulescu died on March 17, 1941.
  Fact 0.2: Nicolae Titulescu was a well-known Romanian diplomat.
  Fact 0.3: Nicolae Titulescu was at various times a government minister.
  Fact 0.4: Nicolae Titulescu was at various times finance minister.
  Fact 0.5: Nicolae Titulescu was at various times foreign minister.
  Fact 0.6: Nicolae Titulescu served for two terms as President of the General Assembly of the League of Nations.
  Fact 0.7: Nicolae Titulescu's two terms as President of the General Assembly of the League of Nations are dated 1930–32.
Sentence 1: The League of Nations (abbreviated as LN in English, "La Société des Nations " ] abbreviated as SDN or SdN in French) was an intergovernmental organisation founded on 10 January 1920 as a result of the Paris Peace Conference that ended the First World War.
  Fact 1.0: The League of Nations is abbreviated as LN in English.
  Fact 1.1: The League of Nations is abbreviated as SDN or SdN in French.
  Fact 1.2: The French name of the League of Nations is 'La Société des Nations'.
  Fact 1.3: The League of Nations was an intergovernmental organisation.
  Fact 1.4: The League of Nations was founded on 10 January 1920.
  Fact 1.5: The League of Nations was founded as a result of the Paris Peace Conference.
  Fact 1.6: The Paris Peace Conference ended the First World War.

Retrieved items:
<item 1>
Nicolae Titulescu
Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister,
and for two terms President of the General Assembly of the League of Nations (1930–32).
</item 1>

<item 2>
Gheorghe Tătărescu
Gheorghe I. Tătărescu (also known as "Guță Tătărescu", with a slightly antiquated pet form of his given name; 2 November 1886 – 28 March 1957) was a Romanian
politician who served 36th Prime Minister of Romania (1934–1937; 1939–1940), three times as Minister of Foreign Affairs ("interim" in 1934 and 1938; appointed
to the office in 1945-1947), and once as Minister of War (1934).

 Representing the "young liberals" faction inside the National Liberal Party (PNL), Tătărescu began his political career as a collaborator of Ion G. Duca,
becoming noted for his anti-Communism and, in time, for his conflicts with the PNL's leader Dinu Brătianu and the Foreign Minister Nicolae Titulescu.
</item 2>

<item 3>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
</item 3>

<item 4>
Remus Nicolai
Remus Nicolai (b. June 10, 1977 in Bistriţa, Romania) is a retired Romanian aerobic gymnast.

 He had a successful career winning six world championships medals (one gold, two silver and three bronze) and five European championships medals (two gold, two
silver and one bronze).

 After retiring from aerobic gymnastics he opened together with his wife (Daniela Mărănducă) a private gymnastics club in Constanţa.

 As of 2013 he trains, together with his wife, the junior artistic women's National Team of Romania at the National Olympic Center in Onesti.
</item 4>

<item 5>
Ioan Silviu Suciu
Ioan Silviu Suciu (born 24 November 1977 in Sibiu) is a retired Romanian artistic gymnast.

 His best event was the pommel horse.

 One of his closest rivals was his team mate and pommel horse Olympic champion Marius Urzică, who was defeated by Suciu in Ljubljana, 2004, when the latter won
the title while Urzică took 8th place.

 Suciu is an Olympic bronze medalist with the team, a silver world medalist on pommel horse and a six-time European medalist (pommel horse, vault and team).

 Suciu was one of the key team members of the Romanian gymnastics team for several years contributing to the 2004 Olympic team bronze medal and three
continental team medals (two gold and one silver).

 He also placed fourth all around at the 2004 Olympic Games.
</item 5>

<item 6>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 6>

<item 7>
Académie Diplomatique Internationale
The Académie Diplomatique Internationale (ADI; english "International Diplomatic Academy") is an international organization based in Paris, France, focused on
modern diplomacy and international affairs.

 Founded in 1926, the ADI was, along with Chatham House in London and the Council on Foreign Relations in New York, one of the first policy institutions devoted
to the sustained study and analysis of international relations.

 Early members included Aristide Briand, Nicolae Titulescu, Gustav Stresemann, and Franklin D. Roosevelt.

 Under the Presidency of His Highness the Aga Khan IV, who was elected in 2000, the ADI has focused its efforts on diplomatic training and emerging dynamics in
international relations and modern diplomacy.
</item 7>

<item 8>
Lippo Hertzka
Lippo Hertzka (19 November 1904 – 14 March 1951) was a Hungarian football player and manager.

 He played for Essener Turnerbund, MTK Budapest and Real Sociedad.

 After retiring, he coached seven teams, including Real Sociedad and Real Madrid, a team which he coached for 2 years (1930–1932) and led to an undefeated La
Liga championship during the 1931-32 season, which meant the first La Liga title for the "white" squad.

 He also won two league titles in Portugal for Benfica.
</item 8>

<item 9>
Gheorghe Tătărescu
During his first time in office, he moved closer to King Carol II, leading an ambivalent policy toward the fascist Iron Guard and ultimately becoming
instrumental in establishing the authoritarian and corporatist regime around the National Renaissance Front.

 In 1940, he accepted the cession of Bessarabia and Northern Bukovina to the Soviet Union, and consequently had to resign.
</item 9>

<item 10>
Ghimpețeni
Ghimpețeni is a commune in Olt County, Romania.

 It is composed of two villages, Ghimpețeni and Ghimpețenii Noi.

 These were part of Nicolae Titulescu Commune until 2004, when they were split off.
</item 10>
=== END ITEM id=df1475aeabb41581 ===

=== ITEM id=a3553a88b00c31f5 ===
Gold sentences and facts:
Sentence 0: Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil War.
  Fact 0.0: Am Rong was born in 1929.
  Fact 0.1: Am Rong died in May 1975.
  Fact 0.2: Am Rong was Cambodian.
  Fact 0.3: Am Rong was a soldier.
  Fact 0.4: Am Rong was a filmmaker.
  Fact 0.5: Am Rong acted as a spokesman on military matters for the Khmer Republic.
  Fact 0.6: Am Rong acted as a spokesman during the Cambodian Civil War.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is American.
  Fact 1.2: Ava Marie DuVernay is a director.
  Fact 1.3: Ava Marie DuVernay is a screenwriter.
  Fact 1.4: Ava Marie DuVernay is a film marketer.
  Fact 1.5: Ava Marie DuVernay is a film distributor.

Retrieved items:
<item 1>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 1>

<item 2>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 2>

<item 3>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 3>

<item 4>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 4>

<item 5>
The entries profile filmmakers Pamela B. Green, Ava DuVernay, Howard Barish, Dana Brown, Matthew O’Neill, and the Maya Deren Award; actors Emayatzy Corinealdi
and Issa Rae; musicians Emel Mathlouthi, Siti binti Saad, and Angélique Kidjo; fitness figures Ingrid Romero and Anais Zanotti; and YouTubers Lilly Singh, Wayne
Goss, and Wengie. They
</item 5>

<item 6>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 6>

<item 7>
Middle of Nowhere (2012 film)
Middle of Nowhere is a 2012 independent feature film written and directed by Ava DuVernay and starring Emayatzy Corinealdi, David Oyelowo, Omari Hardwick and
Lorraine Toussaint.

 The film was the winner of the Directing Award for U.S. Dramatic Film at the 2012 Sundance Film Festival.
</item 7>

<item 8>
Ava DuVernay (born 1972) became the first African-American woman to win Sundance’s directing prize for Middle of Nowhere (2012). Selma (2014) brought historic
Golden Globe and Best Picture Oscar nominations; 13th (2016), examining race and mass incarceration, earned Oscar documentary nominations for DuVernay,
editor-producer Spencer Averick and producer Howard Barish. The entries also cover filmmakers Kathryn Bigelow and Steve McQueen, The
</item 8>

<item 9>
Am Rong
Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil
War.

 Western journalists commented on the irony of his name as he gave briefings which "painted a rosy picture of the increasingly desperate situation on the
ground" during the war.
</item 9>

<item 10>
Emayatzy Corinealdi
Emayatzy Evett Corinealdi ( ; born January 14, 1980) is an American actress.

 She is best known for her leading role in the Ava DuVernay film "Middle of Nowhere" (2012).
</item 10>
=== END ITEM id=a3553a88b00c31f5 ===

=== ITEM id=46eaf71980da468f ===
Gold sentences and facts:
Sentence 0: Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.
  Fact 0.0: Alice Ann Munro's nee name is Laidlaw.
  Fact 0.1: Alice Ann Munro was born on 10 July 1931.
  Fact 0.2: Alice Ann Munro is a Canadian short story writer.
  Fact 0.3: Alice Ann Munro is a Nobel Prize winner.
Sentence 1: Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well as her left-wing sympathies and political activism.
  Fact 1.0: Lillian Florence Hellman was born on June 20, 1905.
  Fact 1.1: Lillian Florence Hellman died on June 30, 1984.
  Fact 1.2: Lillian Florence Hellman was an American dramatist.
  Fact 1.3: Lillian Florence Hellman was an American screenwriter.
  Fact 1.4: Lillian Florence Hellman is known for her success as a playwright on Broadway.
  Fact 1.5: Lillian Florence Hellman is known for her left-wing sympathies.
  Fact 1.6: Lillian Florence Hellman is known for her political activism.

Retrieved items:
<item 1>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 1>

<item 2>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 2>

<item 3>
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 3>

<item 4>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 4>

<item 5>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 5>

<item 6>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 6>

<item 7>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 7>

<item 8>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 8>

<item 9>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 9>

<item 10>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 10>
=== END ITEM id=46eaf71980da468f ===
