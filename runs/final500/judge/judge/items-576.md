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

=== ITEM id=916ff1191840eec4 ===
Gold sentences and facts:
Sentence 0:  At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
  Fact 0.0: She won a silver medal at the 2002 Commonwealth Games.
  Fact 0.1: She won the silver medal with Sara Petersen in the women's doubles at the 2002 Commonwealth Games.
  Fact 0.2: She won a bronze medal in the mixed team at the 2002 Commonwealth Games.
Sentence 1: The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25 July to 4 August 2002.
  Fact 1.0: The 2002 Commonwealth Games were officially known as the XVII Commonwealth Games.
  Fact 1.1: The 2002 Commonwealth Games were commonly known as Manchester 2002.
  Fact 1.2: The 2002 Commonwealth Games were held in Manchester, England.
  Fact 1.3: The 2002 Commonwealth Games were held from 25 July to 4 August 2002.

Retrieved items:
<item 1>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 1>

<item 2>
Nicole Gordon (badminton)
Nicole Gordon (born 17 March 1976 in Lower Hutt, New Zealand) is a female badminton player from New Zealand.

 At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
</item 2>

<item 3>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 3>

<item 4>
Nicole Strausak
Nicole Strausak (born October 21, 1971) is a Swiss curler who retired from competitive curling in March 2005.

 Amongst other awards, she received a World Junior Silver medal in 1991, a European Silver medal in 1993 and a World Silver medal in 2000.

 In 1999, at the European Curling Championships in Chamonix, as well as receiving a Bronze medal, she was awarded the Fair Play & Friendly Award by her fellow
athletes for her conduct and attitude, on and off the ice.
</item 4>

<item 5>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 5>

<item 6>
1999 Copa Colsanitas – Doubles
The 1999 Copa Colsanitas – Doubles was the doubles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Janette Husárová and Paola Suárez were the defending champions but only Suárez competed that year with Laura Montalvo.
</item 6>

<item 7>
2002 Commonwealth Games
The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25
July to 4 August 2002.

 The 2002 Games were to be hosted in the United Kingdom to coincide with the Golden Jubilee of Elizabeth II, head of the Commonwealth, and Manchester was
selected for the 2002 Games ahead of London.

 The XVII Commonwealth Games was, prior to the 2012 Summer Olympics, the largest multi-sport event ever to be held in the UK, eclipsing the London 1948 Summer
Olympics in numbers of teams and athletes participating.

 In terms of sports and events, the 2002 Games were the largest Commonwealth Games in history featuring 281 events across 17 sports.
</item 7>

<item 8>
2006 Generali Ladies Linz – Doubles
Gisela Dulko and Květa Peschke were the defending champions, but competed this year with different partners.

 Dulko teamed up with Michaëlla Krajicek and lost in first round to Eleni Daniilidou and Jasmin Wöhr, while Peschke teamed up with Francesca Schiavone and had
to withdrew in their semifinal match (against Corina Morariu and Katarina Srebotnik) as Schiavone had her right knee injured.
</item 8>

<item 9>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 9>

<item 10>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 10>
=== END ITEM id=916ff1191840eec4 ===

=== ITEM id=af5266ddbc3a99b3 ===
Gold sentences and facts:
Sentence 0: Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera, Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard, Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
  Fact 0.0: Janet Williams is an American soprano.
  Fact 0.1: Janet Williams has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera, Theatre des Champs-Elysees, Opera de Lyon, Nice Opera, Theatre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre.
  Fact 0.2: Janet Williams has performed in concerts throughout Europe, North America, Canada, Israel and Japan.
  Fact 0.3: Janet Williams has performed with conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, Rene Jacobs, Marek Janowski, Neeme Jarvi, Raymond Leppard, Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
Sentence 1: Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized solo pianist, chamber music performer, and conductor.
  Fact 1.0: Vladimir Davidovich Ashkenazy's Russian name is Vladimir Davidovich Ashkenazi.
  Fact 1.1: Vladimir Davidovich Ashkenazy was born on 6 July 1937.
  Fact 1.2: Vladimir Davidovich Ashkenazy is an internationally recognized solo pianist.
  Fact 1.3: Vladimir Davidovich Ashkenazy is an internationally recognized chamber music performer.
  Fact 1.4: Vladimir Davidovich Ashkenazy is an internationally recognized conductor.
Sentence 2:  His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
  Fact 2.0: Vladimir Davidovich Ashkenazy's recordings have earned him five Grammy awards.
  Fact 2.1: Vladimir Davidovich Ashkenazy has received Iceland's Order of the Falcon.

Retrieved items:
<item 1>
Don Williams discography
This is a detailed discography for American country music artist Don Williams.
</item 1>

<item 2>
Fred Hersch
Fred Hersch (born October 21, 1955) is an American jazz pianist and educator.

 He has performed solo and led his own groups, including the Pocket Orchestra consisting of piano, trumpet, voice, and percussion.

 He was the first person to play weeklong engagements as a solo pianist at the Village Vanguard in New York City.

 He has recorded more than 70 of his jazz compositions.

 Hersch has been nominated for several Grammy Awards, and, as of December 2014, had been on the Jazz Studies faculty of the New England Conservatory since 1980
(with breaks).
</item 2>

<item 3>
Chico Hamilton Trio
Chico Hamilton Trio is an album by drummer and bandleader Chico Hamilton recorded at sessions in 1953, 1954 and 1956 released on the Pacific Jazz label.

 The album features Hamilton's first recordings for Pacific Jazz from 1953 and 1954, six tracks originally released on a 10-inch album, along with an additional
four recordings from 1956.
</item 3>

<item 4>
Steve Winwood
Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock,
and jazz.

 Though primarily a vocalist and keyboardist, Winwood also plays bass guitar, drums, acoustic and electric guitar, mandolin, violin, and other strings.
</item 4>

<item 5>
Robert Glasper
Robert Glasper (born April 6, 1978, in Houston, Texas) is an American pianist and record producer.

 He has been nominated for 6 Grammys, has won 3 Grammy Awards and is currently nominated for an Emmy Award.

 His 2012 album "Black Radio" won the Grammy Award for Best R&B Album at the 55th Grammy Awards.

 His 2014 album "Black Radio 2" won the Grammy Award for Best Traditional R&B Performance at the 56th Grammy Awards.

 The song "These Walls" from Kendrick Lamar's album "To Pimp A Butterfly" won Best Rap/Sung Collaboration at the 57th Grammy Awards, on which Glasper plays
keys.

 The soundtrack for the film "Miles Ahead" won Best Soundtrack Compilation at the 58th Grammy Awards, for which Glasper was a producer.

 The song "Letter To The Free", written with Common, is nominated for an Emmy Award for Best Original Song in the Ava Duvernay documentary film "13th" (Netflix)
at the 2017 Emmys.
</item 5>

<item 6>
David M. Arden
David M. Arden (born September 6, 1949) is an American concert pianist whose performing and recording career has focused predominantly on contemporary and
American classical repertoire, including premiere performances and first recordings of piano works by a number of notable contemporary composers, such as Henryk
Górecki, Luciano Berio, Earle Brown, Carson Kievman and David Lang.

 The Russian composer Galina Ustvolskaya named Arden's recording of her 12 Preludes for Piano as her preferred recording of that work.
</item 6>

<item 7>
Vladimir Ashkenazy
Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized
solo pianist, chamber music performer, and conductor.

 He is originally from Russia and has held Icelandic citizenship since 1972.

 He has lived in Switzerland since 1978.

 Ashkenazy has collaborated with well-known orchestras and soloists.

 In addition, he has recorded a large storehouse of classical and romantic works.

 His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
</item 7>

<item 8>
The Impeccable Mr. Wilson
The Impeccable Mr. Wilson is an album by American jazz pianist Teddy Wilson featuring performances recorded in 1956 for the Norgran label.
</item 8>

<item 9>
Janet Williams (soprano)
Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera,
Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco
Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with
conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard,
Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
</item 9>

<item 10>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 10>
=== END ITEM id=af5266ddbc3a99b3 ===

=== ITEM id=766d6c78a89f3074 ===
Gold sentences and facts:
Sentence 0:  The album was Prince's third to be released that year (along with "Chaos and Disorder" and the soundtrack album of the Spike Lee movie "Girl 6"), which made 1996 one of the most prolific years for material released by Prince.
  Fact 0.0: The album was Prince's third album released in 1996.
  Fact 0.1: One of Prince's other 1996 releases was "Chaos and Disorder".
  Fact 0.2: One of Prince's other 1996 releases was the soundtrack album of the Spike Lee movie "Girl 6".
  Fact 0.3: 1996 was one of the most prolific years for material released by Prince.
Sentence 1:  His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
  Fact 1.0: Prince's production company is 40 Acres and a Mule Filmworks.
  Fact 1.1: 40 Acres and a Mule Filmworks has produced over 35 films since 1983.

Retrieved items:
<item 1>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 1>

<item 2>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 2>

<item 3>
Mark Lee Ping-bing
Mark Lee Ping-bing (李屏賓; born August 8, 1954) is a Taiwanese cinematographer, photographer and author with over 70 films and 21 international awards to his
credit including 2 Glory Of The Country Awards from the Government Information Office of Taiwan and the president of Taiwan's Light Of The Cinema Award.

 Lee began his film career in 1977 and in 1985 he started his prolific collaboration with Taiwanese filmmaker Hou Hsiao-hsien.

 Known best for his use of natural lighting utilizing real film and graceful camera movement, Lee received the Grand Technical Prize at the Cannes Film Festival
in 2000 for "In the Mood for Love".

 A member of the Academy of Motion Picture Arts and Sciences, Lee was honored with nominations by the American Society of Cinematographers for its 2014 First
Annual Spotlight Award for Best Cinematography for his work on the 2012 film "Renoir" and by the French Academy of Cinema Arts for a Cesar Award for Best
Cinematography in 2014 also for the film "Renoir".
</item 3>

<item 4>
Spike Lee
Shelton Jackson "Spike" Lee (born March 20, 1957) is an American film director, producer, writer, and actor.

 His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
</item 4>

<item 5>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 5>

<item 6>
Cannibal boom
Only four other cannibal films were made after 1981 until the fad's conclusion in 1988 with Antonio Climati's "Natura Contro".
</item 6>

<item 7>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 7>

<item 8>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 8>

<item 9>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 9>

<item 10>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 10>
=== END ITEM id=766d6c78a89f3074 ===

=== ITEM id=9c939288a7404780 ===
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
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 1>

<item 2>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 2>

<item 3>
Ridley Scott
Sir Ridley Scott (born 30 November 1937) is an English film director and producer.

 Following his commercial breakthrough with the science-fiction horror film "Alien" (1979), his best known works include the neo-noir dystopian science fiction
film "Blade Runner" (1982), historical drama and Best Picture Oscar winner "Gladiator" (2000), and science fiction film "The Martian" (2015).
</item 3>

<item 4>
Andy Warhol
Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual
art movement known as pop art.

 His works explore the relationship between artistic expression, celebrity culture, and advertising that flourished by the 1960s, and span a variety of media,
including painting, silkscreening, photography, film, and sculpture.

 Some of his best known works include the silkscreen paintings "Campbell's Soup Cans" (1962) and "Marilyn Diptych" (1962), the experimental film "Chelsea Girls"
(1966), and the multimedia events known as the "Exploding Plastic Inevitable" (1966–67).
</item 4>

<item 5>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 5>

<item 6>
John Warhola
John Warhola (May 31, 1925 – December 24, 2010) played a pivotal role in maintaining the legacy of his younger brother, pop artist Andy Warhol, assigned
responsibility by their father on his deathbed to ensure that Andy attended college and serving as a trustee of the Andy Warhol Foundation for the Visual Arts
after his brother's death in 1987.

 Warhola oversaw the establishment of The Andy Warhol Museum in Pittsburgh and the Andy Warhol Museum of Modern Art in Medzilaborce, Slovakia.
</item 6>

<item 7>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 7>

<item 8>
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 8>

<item 9>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 9>

<item 10>
The Andy Warhol Story
The Andy Warhol Story (1966) is an underground film directed by Andy Warhol with cinematography by Paul Morrissey, and starring Edie Sedgwick and Rene Ricard
(as Andy Warhol).
</item 10>
=== END ITEM id=9c939288a7404780 ===

=== ITEM id=4f95e2d638c55327 ===
Gold sentences and facts:
Sentence 0:  They competed at the 2015 Pacific Games, where they finished with an 0-4 record.
  Fact 0.0: The unnamed subject competed at the 2015 Pacific Games
  Fact 0.1: The unnamed subject finished the 2015 Pacific Games with an 0-4 record
Sentence 1: Basketball at the 2015 Pacific Games in Port Moresby, Papua New Guinea was held at the BSP Arena and PNG Power Dome on 3–12 July 2015.
  Fact 1.0: Basketball at the 2015 Pacific Games was held in Port Moresby, Papua New Guinea
  Fact 1.1: Basketball at the 2015 Pacific Games was held at the BSP Arena and PNG Power Dome
  Fact 1.2: Basketball at the 2015 Pacific Games was held on 3 to 12 July 2015

Retrieved items:
<item 1>
The 2011 Pan American Games in Guadalajara involved about 6,000 athletes from 42 nations; Toronto hosted the 2015 edition across 18 Golden Horseshoe
communities. Port Moresby’s 2015 Pacific Games featured Kiribati, Australia’s debut, and events including basketball, football and powerlifting. Papua New
Guinea dominated women’s powerlifting; Nauru led the men’s, while Tuvalu earned its first Pacific Games gold. Vanuatu
</item 1>

<item 2>
Guadalajara’s 2011 Pan American Games hosted 6,000 athletes from 42 nations; Toronto’s 2015 Games spanned 18 communities. Port Moresby’s 2015 Pacific Games
included Australia’s debut: Papua New Guinea led women’s powerlifting, Nauru men’s, and Tuvalu won its first gold; Nicolas N’Godrela earned three tennis golds.
Profiles span footballers, champion jockeys (Tony McCoy: 4
</item 2>

<item 3>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 3>

<item 4>
Men's football at the 2015 Pacific Games
The 2015 Pacific Games men's football tournament was the 14th edition of the Pacific Games men's football tournament.

 The men's football tournament was held in Port Moresby, Papua New Guinea between 3–17 July 2015 as part of the 2015 Pacific Games.

 The tournament was age-restricted and open to men's under-23 national teams only.
</item 4>

<item 5>
2015 Pacific Games
The 2015 Pacific Games was held in Port Moresby, Papua New Guinea, from 4 to 18 July 2015.

 It was the fifteenth staging of the Pacific Games.
</item 5>

<item 6>
Basketball at the 2015 Pacific Games
Basketball at the 2015 Pacific Games in Port Moresby, Papua New Guinea was held at the BSP Arena and PNG Power Dome on 3–12 July 2015.
</item 6>

<item 7>
Kiribati national basketball team
The Kiribati national basketball team are the basketball side that represent Kiribati in international competitions.

 They competed at the 2015 Pacific Games, where they finished with an 0-4 record.
</item 7>

<item 8>
Powerlifting at the 2015 Pacific Games
Powerlifting at the 2015 Pacific Games was held from 9–10 July 2015 in the Sir John Guise Indoor Power Dome at Port Moresby, Papua New Guinea.

 The host nation Papua New Guinea was the dominant team, particularly in the women's divisions where it claimed six of the seven possible gold medals.

 Nauru was the strongest team in the men's divisions, winning four of the eight weight categories.

 Telupe Iosefa received the first ever gold medal won by Tuvalu at the Pacific Games in the 120 kg male division.
</item 8>

<item 9>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 9>

<item 10>
Kiribati at the 2015 Pacific Games
Kiribati competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 A total of 86 competitors for Kiribati were listed as of 4 July 2015.
</item 10>
=== END ITEM id=4f95e2d638c55327 ===

=== ITEM id=e9009dd5621158d3 ===
Gold sentences and facts:
Sentence 0: Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.
  Fact 0.0: Dangal's English title is "Wrestling competition".
  Fact 0.1: Dangal is a 2016 film.
  Fact 0.2: Dangal is an Indian film.
  Fact 0.3: Dangal is a Hindi-language film.
  Fact 0.4: Dangal is a biographical sports drama film.
  Fact 0.5: Dangal is loosely based on the story of Mahavir Singh Phogat.
  Fact 0.6: Mahavir Singh Phogat is an amateur wrestler.
  Fact 0.7: Mahavir Singh Phogat trains his daughters Geeta Phogat and Babita Kumari.
  Fact 0.8: Mahavir Singh Phogat trains his daughters to be world-class wrestlers.
Sentence 1: The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
  Fact 1.0: The Man from Snowy River II is a 1988 film.
  Fact 1.1: The Man from Snowy River II is an Australian drama film.
  Fact 1.2: The Man from Snowy River II is the sequel to the 1982 film "The Man from Snowy River".

Retrieved items:
<item 1>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 1>

<item 2>
The Man from Snowy River and Other Verses
The Man from Snowy River and Other Verses (1895) is the first collection of poems by Australian poet Banjo Paterson.

 It was released in hardback by Angus and Robertson in 1895, and features the poet's widely anthologised poems "The Man from Snowy River", "Clancy of the
Overflow", "Saltbush Bill" and "The Man from Ironbark".

 It also contains the poet's first two poems that featured in The Bulletin Debate, a famous dispute in "The Bulletin" magazine from 1892-93 between Paterson and
Henry Lawson.
</item 2>

<item 3>
The Man from Snowy River (poem)
"The Man from Snowy River" is a poem by Australian bush poet Banjo Paterson.

 It was first published in "The Bulletin", an Australian news magazine, on 26 April 1890, and was published by Angus & Robertson in October 1895, with other
poems by Paterson, in "The Man from Snowy River and Other Verses".
</item 3>

<item 4>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 4>

<item 5>
The Man from Snowy River II
The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
</item 5>

<item 6>
The Man from Snowy River (soundtrack)
"The Man from Snowy River is the original motion picture soundtrack from the 1982 film "The Man from Snowy River".
</item 6>

<item 7>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 7>

<item 8>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 8>

<item 9>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 9>

<item 10>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 10>
=== END ITEM id=e9009dd5621158d3 ===

=== ITEM id=99abc07abc594005 ===
Gold sentences and facts:
Sentence 0: The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical creature of the forest, or Rå, in Swedish folklore.
  Fact 0.0: The skogsrå is also called Forest Rå.
  Fact 0.1: The skogsrå is also called Skogsfrun (The Mistress of the Forest).
  Fact 0.2: The skogsrå is also called Skogssnuvan.
  Fact 0.3: The skogsrå is also called Skogsnymfen (Forest Nymph).
  Fact 0.4: The skogsrå is also called Råndan (the Rå).
  Fact 0.5: The skogsrå is also called huldra.
  Fact 0.6: The skogsrå was a mythical creature of the forest, also called Rå, in Swedish folklore.
Sentence 1: A hulder is a seductive forest creature found in Scandinavian folklore.
  Fact 1.0: A hulder is a seductive forest creature.
  Fact 1.1: Hulder creatures are found in Scandinavian folklore.

Retrieved items:
<item 1>
Selkie (disambiguation)
A selkie is a mythological creature found in Faroese, Icelandic, Irish, and Scottish folklore.
</item 1>

<item 2>
Skogsrå
The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical
creature of the forest, or Rå, in Swedish folklore.
</item 2>

<item 3>
Spettekaka
Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also
in Halland.

 It is an important part of the Scanian culinary heritage.

 The name means "cake on a spit", and this describes the method of preparation: it is the Swedish variation on the spit cake.
</item 3>

<item 4>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 4>

<item 5>
Jenny Nyström
Jenny Eugenia Nyström (13 or 15 June 1854 in Kalmar, Sweden – 17 January 1946 in Stockholm) was a painter and illustrator who is mainly known as the person who
created the Swedish image of the "jultomte" on numerous Christmas cards and magazine covers, thus linking the Swedish version of Santa Claus to the gnomes of
Scandinavian folklore.
</item 5>

<item 6>
Scoliopus
Scoliopus, or fetid adderstongue, is a genus of plant within the Liliaceae family consisting of two species, "Scoliopus bigelovii" and "S. hallii."

 Both are found in deep shaded forests, primarily in the coastal counties of the western United States from central California to northern Oregon.

 The name ""Scoliopus"" derives from the Greek words "skolios" and "pous", meaning curved foot, a reference to the shape of the pedicel.

 Taxonomists believe that "Scoliopus" is closely related to "Calochortus", "Prosartes", "Streptopus" and "Tricyrtis", which all have creeping rhizomes as well
as styles that divide at the tip.
</item 6>

<item 7>
Erotikon (1920 film)
Erotikon is a 1920 Swedish romantic comedy film directed by Mauritz Stiller, starring Tora Teje, Karin Molander, Anders de Wahl and Lars Hanson.

 It is based on the 1917 play "A kék róka" by Ferenc Herczeg.

 The story revolves around an entomology professor obsessed with the sexual life of bugs, and his easygoing wife who is courted by two suitors.
</item 7>

<item 8>
Almaz (mythical beast)
The almaz (Chechen: "almaz" or "hun-sag"; Adyghe: "almesti"), roughly translated as various "feral forest-man" or "stone spirit", is a mythical beast that is
considered to be an evil forest creature with magical powers residing in its hair that exists in Chechen and Ingush folk beliefs and Circassian folk beliefs.

 The first "attestation" of it in writing was by a Bavarian captive of the Mongols, but it is present in the national folklore of Chechens and Ingush and of
Circassians.

 The male almaz is said to be hairy and hideous, and have an axe embedded in its chest, while the female is very beautiful with large breasts and golden hair,
and has a "favorite pastime" of dancing naked at night under the moon.

 The almaz is said to have magical powers residing in its hair, but if the hair is removed or even grabbed, it may be rendered helpless.
</item 8>

<item 9>
Hulder
A hulder is a seductive forest creature found in Scandinavian folklore.

 (Her name derives from a root meaning "covered" or "secret".)

 In Norwegian folklore, she is known as huldra ("the [archetypal] hulder", though folklore presupposes that there is an entire Hulder race and not just a single
individual).

 She is known as the skogsrå "forest spirit" or Tallemaja "pine tree Mary" in Swedish folklore, and ulda in Sámi folklore.

 Her name suggests that she is originally the same being as the völva divine figure Huld and the German Holda.
</item 9>

<item 10>
Vörðr
In Norse mythology, a vǫrðr (pl.

 varðir or verðir — "warden," "watcher" or "caretaker") is a warden spirit, believed to follow from birth to death the soul ("hugr") of every person.

 In Old Swedish, the corresponding word is "varþer"; in modern Swedish "vård", and the belief in them remained strong in Scandinavian folklore up until the last
centuries.

 The English word '"wraith" is derived from "vǫrðr", while "" and "warden" are cognates.
</item 10>
=== END ITEM id=99abc07abc594005 ===

=== ITEM id=e61ce626bd94ab34 ===
Gold sentences and facts:
Sentence 0:  It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.
  Fact 0.0: The car described in this sentence was driven by Eddie Irvine.
  Fact 0.1: Eddie Irvine was in his second year with the team.
  Fact 0.2: The car described in this sentence was also driven by Luciano Burti.
  Fact 0.3: Luciano Burti had deputised for Eddie Irvine at the 2000 Austrian Grand Prix.
Sentence 1: Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.
  Fact 1.0: Edmund Irvine Jr. was born on 10 November 1965.
  Fact 1.1: Edmund Irvine Jr. is a retired British racing driver.
  Fact 1.2: Edmund Irvine Jr. is from Northern Ireland.

Retrieved items:
<item 1>
Jaguar R3
The Jaguar R3 is a Formula One racing car with which Jaguar Racing competed in the 2002 Formula One season.

 The car was launched on January 4, 2002.

 It was driven by Eddie Irvine and Pedro de la Rosa, both retained from 2001.
</item 1>

<item 2>
Spyker F1
The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World
Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.

 The change to the Spyker name was accompanied by a switch in racing livery from the red and white previously used by Midland, to an orange and silver
scheme—already seen on the Spyker Spyder GT2-R—orange being the national colour and the auto racing colour of the Netherlands.

 At the end of the 2007 season the team was sold and renamed Force India.
</item 2>

<item 3>
Jordan 195
The Jordan 195 was a Formula One car for the 1995 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 The team never employed a test driver.

 The engine was a Peugeot A10 3.0 V10.

 The team's main sponsor was Total.
</item 3>

<item 4>
1999 British Grand Prix
The 1999 British Grand Prix was a Formula One motor race held on 11 July 1999 at the Silverstone Circuit near Silverstone, England.

 It was the eighth race of the 1999 Formula One season.

 The 60-lap race was won by McLaren driver David Coulthard after he started from third position.

 Eddie Irvine finished second for the Ferrari team and Williams driver Ralf Schumacher came in third.
</item 4>

<item 5>
Jordan 194
The Jordan 194 was a Formula One car for the 1994 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 Kelvin Burt was named as test driver, but his mileage in the car was limited.

 The engine was a Hart 1035 3.5 V10, a developed version of the engine which had proved promising in 1993.

 The team's main sponsor was Sasol, with additional financial support coming from Arisco, the Irish government and other smaller sponsors.

 With driver aids such as traction control and active suspension banned for the 1994 season, the car used a conventional semi-automatic gearbox and pushrod
double wishbone suspension.

 The simpler technical regulations of 1994 seemed to benefit Jordan and the 194 recaptured the level of performance seen in the team's debut season, 1991.

 Indeed, the cut-down airbox and drooping front wing of the 194 recalled the design features of the 1991 car.
</item 5>

<item 6>
Jaguar R2
The Jaguar R2 was the car with which the Jaguar team competed in the Formula One season.

 It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.

 However, the Brazilian was replaced by Pedro de la Rosa after only four Grands Prix.
</item 6>

<item 7>
1997 Japanese Grand Prix
The 1997 Japanese Grand Prix (officially known as the XXIII Fuji Television Japanese Grand Prix) was a Formula One motor race held on 12 October 1997 at the
Suzuka Circuit, Suzuka.

 It was the 16th and penultimate race of the 1997 Formula One season.

 The 53-lap race was won by Michael Schumacher for the Ferrari team after starting from second position.

 Heinz-Harald Frentzen finished second in a Williams, and Eddie Irvine third in the other Ferrari.

 Irvine led much of the race before moving over to assist Schumacher's championship battle by blocking Drivers' Championship leader Jacques Villeneuve.
</item 7>

<item 8>
1983 Volvo International – Doubles
Sherwood Stewart and Ferdi Taygan were the defending champions but they competed with different partners that year, Stewart with Mark Edmondson and Taygan with
Cássio Motta.
</item 8>

<item 9>
1997 Argentine Grand Prix
The 1997 Argentine Grand Prix was a Formula One motor race held at Autódromo Oscar Alfredo Gálvez in Buenos Aires, Argentina on 13 April 1997.

 It was the third race of the 1997 Formula One season and the 600th World Championship event.

 The 72-lap race was won by Williams driver Jacques Villeneuve after he started from pole position.

 Eddie Irvine finished second for the Ferrari team and Jordan driver Ralf Schumacher came in third.
</item 9>

<item 10>
Eddie Irvine
Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.

 He was a Formula One driver between 1993 and 2002, and runner-up in the 1999 World Drivers' Championship, driving for Scuderia Ferrari.
</item 10>
=== END ITEM id=e61ce626bd94ab34 ===

=== ITEM id=6e568fa75488a8ad ===
Gold sentences and facts:
Sentence 0: Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at Naval Air Station Whidbey Island, Washington.
  Fact 0.0: Electronic Attack Squadron 138 (VAQ-138) is also known as the "Yellow Jackets".
  Fact 0.1: Electronic Attack Squadron 138 is an expeditionary EA-18G Growler squadron of the United States Navy.
  Fact 0.2: Electronic Attack Squadron 138 is based at Naval Air Station Whidbey Island, Washington.
Sentence 1: The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.
  Fact 1.0: The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft.
  Fact 1.1: The Boeing EA-18G Growler is a specialized version of the two-seat F/A-18F Super Hornet.

Retrieved items:
<item 1>
VFA-2
Strike Fighter Squadron 2 (VFA-2) also known as the "Bounty Hunters" is a United States Navy F/A-18F Super Hornet strike fighter squadron based at Naval Air
Station Lemoore, California.

 Their tail code is NE and their callsign is "Bullet".

 They are attached to Carrier Air Wing 2 (CVW-2), a composite unit made up of a wide array of aircraft performing a variety of combat and support missions that
deploy aboard the "Carl Vinson" .
</item 1>

<item 2>
VAQ-138
Electronic Attack Squadron 138 (VAQ-138), also known as the "Yellow Jackets", is an expeditionary EA-18G Growler squadron of the United States Navy based at
Naval Air Station Whidbey Island, Washington.

 Their radio callsign is "RAMPAGE".
</item 2>

<item 3>
Graph query row:
subject: Boeing EA-18G Growler
relationship: PART_OF
object: F/A-18F Super Hornet
</item 3>

<item 4>
No. 1 Squadron RAAF
No. 1 Squadron is a Royal Australian Air Force (RAAF) squadron headquartered at RAAF Base Amberley, Queensland.

 Controlled by No. 82 Wing, it is equipped with Boeing F/A-18F Super Hornet multi-role fighters.

 The squadron was formed under the Australian Flying Corps in 1916 and saw action in the Sinai and Palestine Campaigns during World War I.

 It flew obsolete Royal Aircraft Factory B.E.2s, B.E.12s, Martinsyde G.100s and G.102s, as well as Airco DH.6s, Bristol Scouts and Nieuport 17s, before
re-equipping with the R.E.8 in October 1917 and finally the Bristol Fighter in December.

 Its commanding officer in 1917–18 was Major Richard Williams, later known as the "Father of the RAAF".

 Disbanded in 1919, No. 1 Squadron was re-formed on paper as part of the RAAF in 1922, and re-established as an operational unit three years later.
</item 4>

<item 5>
VFA-154
Strike Fighter Squadron 154 (VFA-154), also known as the "Black Knights", is a United States Navy strike fighter squadron stationed at Naval Air Station
Lemoore.

 The Black Knights are an operational fleet squadron flying the F/A-18F Super Hornet.

 They are currently attached to Carrier Air Wing Eleven and deployed aboard the aircraft carrier USS "Nimitz" .

 Their tailcode is NH and their callsign is "Knight".
</item 5>

<item 6>
VAQ-135
Electronic Attack Squadron 135 (VAQ-135), known as the "Black Ravens", is a United States Navy electronic attack squadron that currently operates the EA-18G
Growler carrier-based electronic warfare jet aircraft.

 The squadron is permanently stationed at Naval Air Station Whidbey Island with a radio callsign of ""Thunder"".
</item 6>

<item 7>
No. 82 Wing RAAF
No. 82 Wing is the strike and reconnaissance wing of the Royal Australian Air Force (RAAF).

 It is headquartered at RAAF Base Amberley, Queensland.

 Coming under the control of Air Combat Group, the wing operates F/A-18F Super Hornet multirole fighters and Pilatus PC-9 forward air control aircraft.

 Its units include Nos. 1 and 6 Squadrons, operating the Super Hornet, and No. 4 Squadron, operating the PC-9.
</item 7>

<item 8>
Boeing EA-18G Growler
The Boeing EA-18G Growler is an American carrier-based electronic warfare aircraft, a specialized version of the two-seat F/A-18F Super Hornet.

 The EA-18G replaced the Northrop Grumman EA-6B Prowlers in service with the United States Navy.

 The Growler's electronic warfare capability is primarily provided by Northrop Grumman.

 The EA-18G began production in 2007 and entered operational service with the US Navy in late 2009.

 Australia has also purchased twelve EA-18Gs, which entered service with the Royal Australian Air Force in 2017.
</item 8>

<item 9>
Boeing F/A-18E/F Super Hornet
The Boeing F/A-18E and F/A-18F Super Hornet are twin-engine carrier-capable multirole fighter aircraft variants based on the McDonnell Douglas F/A-18 Hornet.

 The F/A-18E single-seat and F/A-18F tandem-seat variants are larger and more advanced derivatives of the F/A-18C and D Hornet.

 The Super Hornet has an internal 20 mm M61 rotary cannon and can carry air-to-air missiles and air-to-surface weapons.

 Additional fuel can be carried in up to five external fuel tanks and the aircraft can be configured as an airborne tanker by adding an external air refueling
system.
</item 9>

<item 10>
VAQ-209
Electronic Attack Squadron 209 (VAQ-209) is a United States Navy Reserve electronic attack squadron.

 Known as the "Star Warriors", the squadron flies the EA-18G Growler carrier-based electronic warfare jet aircraft.

 Based at Naval Air Station Whidbey Island, WA, it is assigned to the Tactical Support Wing.
</item 10>
=== END ITEM id=6e568fa75488a8ad ===

=== ITEM id=2eb009105fc4466a ===
Gold sentences and facts:
Sentence 0: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 0.0: Henry Louis Mencken lived from September 12, 1880 to January 29, 1956.
  Fact 0.1: Henry Louis Mencken was an American journalist.
  Fact 0.2: Henry Louis Mencken was an American satirist.
  Fact 0.3: Henry Louis Mencken was an American cultural critic.
  Fact 0.4: Henry Louis Mencken was a scholar of American English.
Sentence 1: Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.
  Fact 1.0: Albert Camus lived from 7 November 1913 to 4 January 1960.
  Fact 1.1: Albert Camus was a French philosopher.
  Fact 1.2: Albert Camus was an author.
  Fact 1.3: Albert Camus was a journalist.

Retrieved items:
<item 1>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 1>

<item 2>
Reflections on the Guillotine
"Reflections on the Guillotine" is an extended essay written in 1957 by Albert Camus.

 In the essay Camus takes an uncompromising position for the abolition of the death penalty.

 Camus's view is similar to that of Cesare Beccaria and the Marquis de Sade, the latter having also argued that murder premeditated and carried out by the state
was the worst kind.

 Camus states that he does not base his argument on sympathy for the convicted but on logical grounds and on proven statistics.

 Camus also argues that capital punishment is an easy option for the government where remedy and reform may be possible.
</item 2>

<item 3>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 3>

<item 4>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 4>

<item 5>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 5>

<item 6>
The Stranger (novel)
L’Étranger (The Outsider [UK], or The Stranger [US]) is a 1942 novel by French author Albert Camus.

 Its theme and outlook are often cited as examples of Camus' philosophy of the absurd and existentialism, though Camus personally rejected the latter label.
</item 6>

<item 7>
The Guest
"The Guest" (French: "L'Hôte" ) is a short story by the French writer Albert Camus.

 It was first published in 1957 as part of a collection entitled "Exile and the Kingdom" ("L'exil et le royaume").

 The French title "L'Hôte" translates into both "the guest" and "the host" which ties back to the relationship between the main characters of the story.

 Camus employs this short tale to reflect upon issues raised by the political situation in French North Africa.

 In particular, he explores the problem of refusing to take sides in the colonial conflict in Algeria, something that mirrors Camus' own non-aligned stance
which he had set out in his Nobel Prize acceptance speech.
</item 7>

<item 8>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 8>

<item 9>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 9>

<item 10>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 10>
=== END ITEM id=2eb009105fc4466a ===

=== ITEM id=cfe62b4bd2c97509 ===
Gold sentences and facts:
Sentence 0: Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.
  Fact 0.0: Awolnation is an American alternative rock band.
  Fact 0.1: Awolnation was formed by Aaron Bruno.
  Fact 0.2: Awolnation is fronted by Aaron Bruno.
  Fact 0.3: Aaron Bruno was formerly of Under the Influence of Giants.
  Fact 0.4: Aaron Bruno was formerly of Home Town Hero.
  Fact 0.5: Aaron Bruno was formerly of Insurgence.
Sentence 1: This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.
  Fact 1.0: This Mortal Coil was a music collective.
  Fact 1.1: This Mortal Coil was led by Ivo Watts-Russell.
  Fact 1.2: Ivo Watts-Russell founded the British record label 4AD.

Retrieved items:
<item 1>
This Mortal Coil
This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.

 Although Watts-Russell and John Fryer were technically the only two official members, the band's recorded output featured a large rotating cast of supporting
artists, many of whom were otherwise associated with 4AD, including members of Cocteau Twins, Colourbox, and Dead Can Dance.

 The project became known for its gothic, dream pop sound, and released three full albums beginning in 1984 with "It'll End in Tears".
</item 1>

<item 2>
Awolnation
Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.
</item 2>

<item 3>
Not Your Fault
"Not Your Fault" is a song by American alternative rock band Awolnation.

 It was originally written by frontman Aaron Bruno and recorded by the band for their debut studio album "Megalithic Symphony", where it appears as the twelfth
track.

 "Not Your Fault" was released to American modern rock radio on October 17, 2011, and was also released as a single in the United Kingdom and Bosnia on February
26, 2012.

 The single peaked at numbers three and eleven on the US "Billboard" Alternative Songs and Rock Songs charts, respectively.
</item 3>

<item 4>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 4>

<item 5>
Graph query row:
entity: This Mortal Coil
type: music collective
</item 5>

<item 6>
Hollow Moon (Bad Wolf)
"Hollow Moon (Bad Wolf)" is a song by American alternative rock band Awolnation.

 It was released as the lead single from their second studio album "Run" (2015).

 The song became their first number-one hit on the Billboard Alternative Songs chart in early 2015.
</item 6>

<item 7>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 7>

<item 8>
Graph query row:
entity: Awolnation
type: American alternative rock band
</item 8>

<item 9>
Kill Your Heroes
"Kill Your Heroes" is a song by American alternative rock band Awolnation.

 It is written by lead singer Aaron Bruno and musician Brian West for the band's debut studio album "Megalithic Symphony", where it appears as the eighth track.

 "Kill Your Heroes" was released as the third and final single from "Megalithic Symphony" and reached the top 20 of the United States "Billboard" Alternative
Songs and Rock Songs charts.
</item 9>

<item 10>
Awolnation discography
American alternative rock band Awolnation has released two studio albums, three extended plays, eight singles and ten music videos.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.

 The band's second studio album, "Run", was released on March 17, 2015.
</item 10>
=== END ITEM id=cfe62b4bd2c97509 ===

=== ITEM id=26eb4736677a7210 ===
Gold sentences and facts:
Sentence 0: The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
  Fact 0.0: The Second Battle of Guam was fought from 21 July to 10 August 1944.
  Fact 0.1: The Second Battle of Guam was the American recapture of the Japanese-held island of Guam.
  Fact 0.2: Guam is a U.S. territory in the Mariana Islands.
  Fact 0.3: Japan captured Guam from the U.S. in the 1941 First Battle of Guam.
  Fact 0.4: The Second Battle of Guam took place during the Pacific campaign of World War II.
Sentence 1: The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and Antietam Creek as part of the Maryland Campaign.
  Fact 1.0: The Battle of Antietam was also known as the Battle of Sharpsburg, particularly in the South.
  Fact 1.1: The Battle of Antietam was fought on September 17, 1862.
  Fact 1.2: The Battle of Antietam was fought near Sharpsburg, Maryland.
  Fact 1.3: The Battle of Antietam was fought near Antietam Creek.
  Fact 1.4: The Battle of Antietam was part of the Maryland Campaign.

Retrieved items:
<item 1>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 1>

<item 2>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 2>

<item 3>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 3>

<item 4>
Maryland Campaign
The Maryland Campaign—or Antietam Campaign—occurred September 4–20, 1862, during the American Civil War.

 Confederate Gen. Robert E. Lee's first invasion of the North was repulsed by the Army of the Potomac under Maj. Gen. George B. McClellan, who moved to
intercept Lee and his Army of Northern Virginia and eventually attacked it near Sharpsburg, Maryland.

 The resulting Battle of Antietam was the bloodiest single-day battle in American history and is widely considered one of the major turning points of the war.
</item 4>

<item 5>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 5>

<item 6>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 6>

<item 7>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 7>

<item 8>
William K. MacNulty
William Kirk MacNulty (May 22, 1892 - August 3, 1964) was a U. S. Marine with a long and distinguished record.

 He was a young U.S. Marine Corps Second Lieutenant during World War One and saw action at the Battle of the Argonne Forest.

 He served as a captain during the Second U.S. Nicaraguan Campaign (1926–33).

 During the Second World War as Lieutenant Colonel he commanded the U.S. Marine Corps defense of Guam against overwhelming Imperial Japanese forces during the
First Battle of Guam.

 He was incarcerated by the Japanese as a prisoners of war.

 He was promoted to Brigadier General during captivity and retired from military service in 1946.

 He is buried at the Golden Gate National Cemetery in San Bruno, San Mateo County, California.
</item 8>

<item 9>
Battle of Antietam
The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and
Antietam Creek as part of the Maryland Campaign.

 It was the first field army–level engagement in the Eastern Theater of the American Civil War to take place on Union soil and is the bloodiest single-day
battle in American history, with a combined tally of 22,717 dead, wounded, or missing.
</item 9>

<item 10>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 10>
=== END ITEM id=26eb4736677a7210 ===
