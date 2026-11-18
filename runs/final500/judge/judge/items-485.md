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

=== ITEM id=35a4df9ec961c38c ===
Gold sentences and facts:
Sentence 0: William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.
  Fact 0.0: William Dwight Schultz was born on November 24, 1947.
  Fact 0.1: William Dwight Schultz is an American actor.
  Fact 0.2: William Dwight Schultz is an American voice artist.
Sentence 1:  He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the Squirrel in "CatDog".
  Fact 1.0: William Dwight Schultz is well known in animation as the mad scientist Dr. Animo in the "Ben 10" series.
  Fact 1.1: William Dwight Schultz is well known in animation as Chef Mung Daal in the children's cartoon "Chowder".
  Fact 1.2: William Dwight Schultz is well known in animation as Eddie the Squirrel in "CatDog".
Sentence 2: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 2.0: Chowder is an American animated television series.
  Fact 2.1: Chowder was created by C. H. Greenblatt.
  Fact 2.2: Chowder was created for Cartoon Network.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Dan Castellaneta
Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.

 Noted for his long-running role as Homer Simpson on the animated television series "The Simpsons", he also voices many other characters for the show, including
Abraham "Grampa" Simpson, Barney Gumble, Krusty the Clown, Sideshow Mel, Groundskeeper Willie, Mayor Quimby and Hans Moleman.
</item 2>

<item 3>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 3>

<item 4>
Rodger Bumpass
Rodger Bumpass (born November 20, 1951) is an American actor and voice actor.

 He is best known for his long-running role as Squidward Tentacles on the American animated television series "SpongeBob SquarePants".

 He voices many other characters on the show, including Dr. Forrest and various anchovies.

 He also voiced The Chief in the animated series "Where on Earth Is Carmen Sandiego?

", and Mr. Besser, the school principal in the animated series "The Kids from Room 402".

 Bumpass has many other credits in animated films, animated television series, and video games.
</item 4>

<item 5>
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
</item 5>

<item 6>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 6>

<item 7>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 7>

<item 8>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 8>

<item 9>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 9>

<item 10>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 10>
=== END ITEM id=35a4df9ec961c38c ===

=== ITEM id=756cddc6520548bd ===
Gold sentences and facts:
Sentence 0: Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National Remembrance.
  Fact 0.0: Kolejka is a board game.
  Fact 0.1: Kolejka is Polish for 'queue' or 'line (of people)'.
  Fact 0.2: Karol Madaj designed Kolejka.
  Fact 0.3: Kolejka was launched in February 2011.
  Fact 0.4: Kolejka was launched by Poland's Institute of National Remembrance.
Sentence 1: Ghettopoly is a parody of "Monopoly" released in 2003.
  Fact 1.0: Ghettopoly is a parody of Monopoly.
  Fact 1.1: Ghettopoly was released in 2003.

Retrieved items:
<item 1>
Plaid Hat Games
Plaid Hat Games is a United States-based board game studio.

 Plaid Hat Games was founded in 2009.

 Board game designer Colby Dauch formed a board game publishing company in order to release the companies first game, Summoner Wars.
</item 1>

<item 2>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 2>

<item 3>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 3>

<item 4>
Xeko
Xeko is a collectible card game revolving around endangered species.

 It was launched on Earthday 2006.

 It won the "Creative Child Magazine" 2006 Toy of the Year Award and the National Parenting Center's Seal of Approval in its first year.

 Four "Mission" sets have been released.

 "Mission: Costa Rica" and "Mission: Madagascar", based on biodiversity hotspots were released first.

 "Mission: Indonesia", was released in 2007, with the final release, "Mission: China", was released July 19, 2008.

 A total of thirty more missions were planned but never developed.
</item 4>

<item 5>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 5>

<item 6>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 6>

<item 7>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 7>

<item 8>
Ghettopoly
Ghettopoly is a parody of "Monopoly" released in 2003.

 Invented by David Chang, it uses "Monopoly"-like mechanics in the atmosphere of a caricaturized United States ghetto.
</item 8>

<item 9>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 9>

<item 10>
Egyptian Ratscrew
Egyptian Ratscrew (also known as Egyptian Ratkiller, Egyptian War, and by other names) is a card game of the matching family of games.

 The game is similar to the 19th century British card game Beggar-My-Neighbour, with the added concept of "slapping" cards when certain combinations are played,
similar to and perhaps borrowed from Slapjack.
</item 10>
=== END ITEM id=756cddc6520548bd ===

=== ITEM id=3d06e365a212ee16 ===
Gold sentences and facts:
Sentence 0: Pravda (Russian: Правда ; ] , "Truth") is a Russian broadsheet newspaper, formerly the official newspaper of the Communist Party of the Soviet Union, when it was one of the most influential papers in the country with a circulation of 11 million.
  Fact 0.0: Pravda's Russian name is Правда, which means 'Truth'.
  Fact 0.1: Pravda is a Russian broadsheet newspaper.
  Fact 0.2: Pravda was formerly the official newspaper of the Communist Party of the Soviet Union.
  Fact 0.3: When Pravda was the official newspaper of the Communist Party of the Soviet Union, it was one of the most influential papers in the country.
  Fact 0.4: When Pravda was one of the most influential papers in the country, its circulation was 11 million.
Sentence 1: Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.
  Fact 1.0: Stiffelio is an opera.
  Fact 1.1: Stiffelio is an opera in three acts.
  Fact 1.2: Stiffelio is by Giuseppe Verdi.
  Fact 1.3: Stiffelio uses an Italian libretto by Francesco Maria Piave.

Retrieved items:
<item 1>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 1>

<item 2>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 2>

<item 3>
The entries span opera, music, theatre, literature and Renaissance history. Bernhard Bötel sang roles from Mozart to Verdi; operas range from Gluck and Hasse to
Rossini and later composers. Commedia dell’arte’s masked stock characters, improvisation and lazzi are described. Musical highlights include Tchaikovsky’s
Second Piano Concerto, Debussy’s *Masques* and Bartók’s quartets. Other topics include Verdi’s Busseto theatre,
</item 3>

<item 4>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 4>

<item 5>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 5>

<item 6>
Bernhard Bötel’s roles ranged from Mozart’s Belmonte to Verdi’s Duke of Mantua. The entries survey operas across eras: Gluck’s reform works *Orfeo ed Euridice*
(1762) and *Paride ed Elena* (1770); Hasse’s *Romolo ed Ersilia* (1765); Rossini’s *Mosè in Egitto* (1818); and later works by Donizetti, Bernstein,
</item 6>

<item 7>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 7>

<item 8>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 8>

<item 9>
The material surveys Italian opera history and related figures. Verdi’s Busseto theatre opened in 1868, seats 300, and occupies the former Pallavicino fortress.
Verdi studied under Ferdinando Provesi from age 11; French composer Ambroise Thomas wrote *Mignon* and *Hamlet*, whose 1868 premiere featured bass Armand
Castelmary, later famed for dying onstage in New York. Monteverdi, a Renaissance-to-Baroque pioneer,
</item 9>

<item 10>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 10>
=== END ITEM id=3d06e365a212ee16 ===

=== ITEM id=a50ca7227accbe28 ===
Gold sentences and facts:
Sentence 0:  The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.
  Fact 0.0: The first edition was written by J. Eric Holmes.
  Fact 0.1: The first edition was based on the original work of Gary Gygax and Dave Arneson.
Sentence 1: John Eric Holmes M.D. (February 16, 1930 – March 20, 2010) was an associate professor of neurology at the University of Southern California School of Medicine, an author and promoter of fantasy role-playing games, a noted fan and enthusiast of Edgar Rice Burroughs, and an American writer of non-fiction, fantasy and science fiction.
  Fact 1.0: John Eric Holmes M.D. was born on February 16, 1930.
  Fact 1.1: John Eric Holmes M.D. died on March 20, 2010.
  Fact 1.2: John Eric Holmes M.D. was an associate professor of neurology at the University of Southern California School of Medicine.
  Fact 1.3: John Eric Holmes M.D. was an author and promoter of fantasy role-playing games.
  Fact 1.4: John Eric Holmes M.D. was a noted fan and enthusiast of Edgar Rice Burroughs.
  Fact 1.5: John Eric Holmes M.D. was an American writer of non-fiction, fantasy and science fiction.

Retrieved items:
<item 1>
Dungeons &amp; Dragons Companion Set
The Dungeons & Dragons Companion Set is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1984 as an expansion to the "Dungeons & Dragons Basic Set".
</item 1>

<item 2>
GURPS Basic Set
GURPS Basic Set is a hard-bound two volume set written by Steve Jackson, Sean M. Punch, and David L. Pulver.

 The first edition "GURPS Basic Set" was published in 1986.
</item 2>

<item 3>
John Eric Holmes
John Eric Holmes M.D. (February 16, 1930 – March 20, 2010) was an associate professor of neurology at the University of Southern California School of Medicine,
an author and promoter of fantasy role-playing games, a noted fan and enthusiast of Edgar Rice Burroughs, and an American writer of non-fiction, fantasy and
science fiction.

 His father, Wilfred "Jasper" Holmes was also a writer of adventure stories under the pen name Alec Hudson.

 His writings appeared under his full name and under variants such as Eric Holmes and J. Eric Holmes and the pen name Sidney Leland.
</item 3>

<item 4>
The Keep on the Borderlands
The Keep on the Borderlands is a "Dungeons & Dragons" module by Gary Gygax, first printed in December 1979.

 In it, player characters are based at a keep and investigate a nearby series of caves that are filled with a variety of monsters.

 It was designed to be used with the "Dungeons & Dragons Basic Set", and was included in the 1979–1982 editions of the "Basic Set".

 It was designed for people new to "Dungeons & Dragons".
</item 4>

<item 5>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 5>

<item 6>
Dungeons &amp; Dragons Game (1991 boxed set)
The New Easy-to-Master Dungeons & Dragons Game is an introductory set for the "Dungeons & Dragons" fantasy role-playing game, published by TSR, Inc. in 1991.

 It was a replacement for the previous "Dungeons & Dragons Basic Set", serving to introduce new players to the game, using the rule set previously established.
</item 6>

<item 7>
Dungeons &amp; Dragons Basic Set
The Dungeons & Dragons Basic Set is a set of rulebooks for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 First published in 1977, it saw a handful of revisions and reprintings.

 The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.

 Later editions were edited by Tom Moldvay, Dave Cook, and Frank Mentzer.
</item 7>

<item 8>
Dungeons &amp; Dragons Master Rules
Dungeons & Dragons Master Rules is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1985 as an expansion to the "Basic Set".
</item 8>

<item 9>
Dungeons &amp; Dragons Immortals Rules
Dungeons & Dragons Immortals Rules, written by Frank Mentzer, is a boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game first published by
TSR in 1986 as an expansion to the "Basic Set".
</item 9>

<item 10>
The Lost City (Dungeons &amp; Dragons)
The Lost City (B4) is a "Dungeons & Dragons" adventure module by Tom Moldvay.

 It was first published by TSR in 1982 and was designed as a stand-alone adventure for use with the "Dungeons & Dragons Basic Set".

 The working title for the module was "The Lost City of Cynidecia".

 Moldvay designed the module to give novice Dungeon Masters experience fleshing out adventures and is only partially complete.

 The module is described as a low-level scenario, in which the only hope of the player characters' survival can be found in a ruined city slowly rising out of
the sands.

 The adventure is set inside a huge step pyramid, with the lower pyramid only sketched out and the city itself described with a list of the major areas and a
map.

 The adventure’s main villain is Zargon, a giant one-eyed monster and his minions.

 The entire double pyramid, not including the city, contains over 100 rooms.
</item 10>
=== END ITEM id=a50ca7227accbe28 ===

=== ITEM id=7c02afca58691bf5 ===
Gold sentences and facts:
Sentence 0: The University of Delaware (colloquially "UD") is the largest university in Delaware.
  Fact 0.0: The University of Delaware is colloquially called 'UD'.
  Fact 0.1: The University of Delaware is the largest university in Delaware.
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen's Hungarian name is 'Debreceni Egyetem'.
  Fact 1.1: The University of Debrecen is a university.
  Fact 1.2: The University of Debrecen is located in Debrecen, Hungary.

Retrieved items:
<item 1>
Pike Creek, Delaware
Pike Creek is a census-designated place (CDP) in New Castle County, Delaware and is part of the Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metropolitan
statistical area.

 In 2007, CNN's Money Magazine ranked Pike Creek on its list of the 100 Best Places to Live in the United States.

 Pike Creek was the only community in Delaware to appear on the list.

 The population was 7,898 at the 2010 census.

 Local attractions include Middle Run Natural Area, White Clay Creek State Park, Carousel Farm Park & Equestrian Center, Golf and Restaurants.

 Pike Creek is approximately 5 miles from downtown Newark, Delaware which contains the University of Delaware.

 Pike Creek is connected to the University of Delaware campus and the Newark, Delaware downtown area by several bike trails.

 Pike Creek is approximately 10 miles west of the state's largest city, Wilmington, Delaware which is a national financial center.

 Chester County, Pennsylvania is located approximately 2 miles west of Pike Creek.
</item 1>

<item 2>
Delaware–William &amp; Mary football rivalry
The Delaware–William & Mary football rivalry between the Delaware Fightin' Blue Hens and the William & Mary Tribe is a match-up between two public universities,
the University of Delaware and the College of William and Mary, that are also members of the Colonial Athletic Association.

 Both schools have academic reputations that have labeled them as Public Ivies.

 Both schools are also Colonial Colleges having been founded before the United States became independent in 1776; the College of William and Mary was founded in
1693 and the University of Delaware's predecessor school was founded in 1743.
</item 2>

<item 3>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 3>

<item 4>
The text profiles diverse educational and related institutions worldwide. Highlights include Berat University (Albania), founded in 2009 and expanding to four
faculties; Rotman (Toronto), established in 1950; Auburn, Arizona, Debrecen, Copenhagen, Dalhousie and other research universities; specialist schools such as
Falmouth and Aalborghus; and private or state colleges across Ghana, Malaysia, Norway, Pakistan and elsewhere. It also covers Syracuse sororities, a
</item 4>

<item 5>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 5>

<item 6>
Rutgers University–Camden
Rutgers University–Camden is one of three regional campuses of Rutgers University, New Jersey's public research university.

 It is located in Camden, New Jersey, United States.

 Founded in the 1920s, Rutgers–Camden began as an amalgam of the South Jersey Law School and the College of South Jersey.

 It is the southernmost of the three regional campuses of Rutgers—the others being located in New Brunswick and Newark.

 The city of Camden is located on the Delaware River, east of Philadelphia.
</item 6>

<item 7>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 7>

<item 8>
University of Delaware Botanic Gardens
The University of Delaware Botanic Gardens are botanical gardens and an arboretum located on the campus of the University of Delaware, in Newark, Delaware,
United States.

 The gardens are open to the public without charge.
</item 8>

<item 9>
Ellendale, Delaware
Ellendale is a town in Sussex County, Delaware, United States.

 The population was 381 at the 2010 census, an increase of 16.5% since 2000.

 It is part of the Salisbury, Maryland-Delaware Metropolitan Statistical Area.

 Ellendale is the "Gateway to Delaware's Resort Beaches" because it is the town located on U.S. Highway 113, the resort area's westernmost border, and Delaware
Route 16, the resort area's northernmost border with the eastern border being the Delaware Bay and Atlantic Ocean and the southern border being the state line
with Maryland.

 Ellendale is home to the Philadelphia Bible College and the Harbor Christian Academy.
</item 9>

<item 10>
College Park, Delaware
College Park is an unincorporated community in New Castle County, Delaware, United States.

 College Park is located along Delaware Route 896 southwest of Newark.
</item 10>
=== END ITEM id=7c02afca58691bf5 ===

=== ITEM id=198af0b2618fd286 ===
Gold sentences and facts:
Sentence 0: Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.
  Fact 0.0: Corn Ranch is a spaceport.
  Fact 0.1: Corn Ranch is located in the West Texas town of Van Horn, Texas.
  Fact 0.2: Flight tests of the New Shepard are carried out at Corn Ranch.
  Fact 0.3: Flight tests of the New Shepard are carried out by Blue Origin.
Sentence 1: Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its headquarters in Kent, Washington.
  Fact 1.0: Blue Origin is an American privately funded aerospace manufacturer.
  Fact 1.1: Blue Origin is a spaceflight services company.
  Fact 1.2: Blue Origin was set up by Amazon.com founder Jeff Bezos.
  Fact 1.3: Blue Origin has its headquarters in Kent, Washington.

Retrieved items:
<item 1>
Lubbock Arts Festival
The Lubbock Arts Festival is an annual arts festival held in April at the Lubbock Memorial Civic Center in Lubbock, Texas.

 The festival is the largest fine arts and crafts show in West Texas.

 Since its establishment in 1978, the Lubbock Arts Festival focuses on visual, performing, culinary, and children’s art.

 The 2013 Lubbock Arts Festival drew a record attendance of over 30,000.

 In 2014, the Silent Wings Museum participated in the festival.
</item 1>

<item 2>
Petroleum Building (Midland)
The Petroleum Building (formerly known as the Hogan Building) is a highrise in downtown Midland, TX.

 The building was built in 1928 and consists of 12 floors and has a neo-gothic style architecture to it.

 The building stands at 137 ft but with its spires reaches a height of 151 ft. The Hogan building is a registered, Texas historical landmark.

 The tower is named for lawyer and oil entrepreneur Thomas Stephen Hogan.

 For information on Hogan, see article on Jacob Bunn.

 The design of the Petroleum Building, like the Woolworth Building in New York City and the Baum Building in Oklahoma City, utilizes many classic architectural
devices.

 Gothic spires and Moorish arches, lavish carved surfaces and opulent marble, all intended to clothe the business house with the respectability of a cathedral.

 Thomas Stephen Hogan intended his building to be a landmark and (Fort Worth architect-engineer) Wyatt Hedrick designed a building that people would talk about.

 However, behind the ornate cast-stone façade was (sic) functional reinforced concrete, the finest equipment available, and the determination of one man to
establish Midland as the headquarters of the West Texas oil fields.
</item 2>

<item 3>
SyberJet Aircraft
SyberJet Aircraft is the manufacturer of the SJ30 light business jet.

 The company's headquarters is in Cedar City, Utah adjacent to the Cedar City Regional Airport with additional engineering offices and manufacturing, service,
repair and fatigue test facilities near and on the San Antonio International Airport in San Antonio, Texas.
</item 3>

<item 4>
Northern California TRACON
Northern California TRACON (NCT) (Terminal Radar Approach Control), or NorCal TRACON for short, is an air traffic control facility that provides safety alerts,
separation, and sequencing of air traffic arriving, departing, and transiting the airspace and airports in Northern California.

 Located in Rancho Cordova near Sacramento, NCT controls airspace over 19000 square miles, and serves Reno International Airport, Sacramento International
Airport, San Jose International Airport, Oakland International Airport, and San Francisco International Airport, plus 19 other smaller airports with air traffic
control towers.

 NCT is the 3rd busiest TRACON in America.

 NorCal TRACON is the step between local control (in an airport's control tower) and Air Route Traffic Control Center (ARTCC), in this case, Oakland Center
(ICAO code: ZOA).

 San Francisco International Airport is the 2nd largest airport in California and the largest airport serving Northern California.
</item 4>

<item 5>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 5>

<item 6>
H-T Ranch
H-T Ranch is a historic ranch complex 10 mi west of Amidon, North Dakota.

 The ranch originally consisted of ten buildings; however, only the ranch house and log barn survive.

 The ranch served as the headquarters of the Little Missouri Cattle Company, which was run by Arthur Clark Hidekoper.

 Hidekober established the ranch in the 1880s, and by the end of the decade, it had become the "most notable" ranch in the area.

 The ranch house, called Shackford, was built in 1896; its uncommon style resembles a bungalow but also borrows from other styles such as the Stick style.

 By 1906, the "Fargo Forum" described the ranch as "the biggest and most important [ranch] in the state" and "one of the largest horse raising outlets in the
world".

 In the same year, Hidekoper sold the 70,000 acre ranch; the sale was the largest land deal in North Dakota history.

 After the sale, a land company reduced the ranch to 5000 acre ; it was later used as a dude ranch in the 1920s.
</item 6>

<item 7>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 7>

<item 8>
M7 Aerospace
M7 Aerospace LP is an aerospace company with its headquarters on the property of San Antonio International Airport in Uptown San Antonio, Texas, United States.
</item 8>

<item 9>
Blue Origin
Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its
headquarters in Kent, Washington.

 The company is developing technologies to enable private human access to space with the goal to dramatically lower costs and increase reliability.

 Blue Origin is employing an incremental approach from suborbital to orbital flight, with each developmental step building on its prior work.

 The company motto is ""Gradatim Ferociter"", Latin for "Step by Step, Ferociously".

 Blue Origin is developing a variety of technologies, with a focus on rocket-powered Vertical Takeoff and Vertical Landing (VTVL) vehicles for access to
suborbital and orbital space.

 The company's name refers to the blue planet, Earth, as the point of origin.
</item 9>

<item 10>
Corn Ranch
Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.

 The 165,000-acre (670 km²) land parcel was purchased by Internet billionaire Jeff Bezos.

 The first flight test took place on November 13, 2006 with the goal of providing commercial tourist flights.
</item 10>
=== END ITEM id=198af0b2618fd286 ===

=== ITEM id=68933738fe4b249b ===
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
Old Church of St George, Hill Street Dublin
The Old Church of St. George, commonly called "Little George's" in Hill Street (formerly Temple Street Lower) Parish of St. Mary, Dublin was built in 1668 by
the Eccles family for their workmen and also as a chapel-of-ease to the nearby St. Mary's Church.

 However, that St. Mary's Church was not St. Mary's Church, Dublin as its foundation stone was laid in 1700, and it was not St. Mary's Abbey, Dublin as it was
dissolved in 1539.

 Therefore St George's church, Hill Street, may have been a Chapel-of-Ease to St. Michan's Church in Church Street.

 The main body of the church, with the exception of the tower, was demolished in 1894.
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
St. Mary's Church (Ballston Spa, New York)
St. Mary's Church is a Catholic parish located in Ballston Spa, New York.

 It is located within the Roman Catholic Diocese of Albany.

 Father Thomas J. Kelly is the current pastor.

 St. Mary's is the fourth oldest parish in the Diocese.

 St. Mary's of Ballston Spa is partnered with St. Mary's of Galway, after the pastor at St. Mary's of Galway died and no replacement was available.
</item 3>

<item 4>
Graph query row:
entity: Monastery of the Mother of God at the Spring
entity_kind: building
country: None
location: None
relationship: LOCATED_IN
connected_entity: Istanbul
connected_kind: city
connected_country: Turkey
connected_location: None
held_on: None
shared_connection: True
</item 4>

<item 5>
Graph query row:
entity: Hagia Irene
entity_kind: Greek Eastern Orthodox church
country: None
location: None
relationship: LOCATED_IN
connected_entity: Istanbul
connected_kind: city
connected_country: Turkey
connected_location: None
held_on: None
shared_connection: True
</item 5>

<item 6>
St. Mary's Episcopal Church (Washington, D.C.)
St. Mary's Episcopal Church, also known as St. Mary's, Foggy Bottom or St. Mary's Chapel, is a historic Episcopal church located at 730 23rd Street, N.W. in the
Foggy Bottom neighborhood of Washington, D.C..

 On April 2, 1973, St. Mary's Episcopal Church was added to the National Register of Historic Places.
</item 6>

<item 7>
St Mary de Lode Church, Gloucester
St Mary de Lode Church, Archdeacon Street, Gloucester GL1 2QT, is a Church of England church immediately outside the grounds of Gloucester Cathedral.

 It is believed by some to be on the site of the first Christian church in Britain.

 The church is in the Diocese of Gloucester and Grade I listed by English Heritage.

 It has also been known as "St. Mary Before the Gate of St. Peter", "St. Mary Broad Gate" and "St. Mary De Port".
</item 7>

<item 8>
Church of St. Mary of the Spring (Istanbul)
The Monastery of the Mother of God at the Spring (full name in Greek: , pr.

 "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an
Eastern Orthodox sanctuary in Istanbul, Turkey.

 The present church, built in 1835, bears the same dedication as the shrine erected in this place between the end of the fifth and the beginning of the sixth
century.

 After several renovations, this building was destroyed in the first half of the fifteenth century by the Ottomans.

 The complex got its name from a nearby holy spring, reputed to have healing properties.

 For almost fifteen hundred years, this sanctuary has been one of the most important pilgrimage sites of Greek Orthodoxy.
</item 8>

<item 9>
Hagia Irene
Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek
Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.

 It is one of the few churches in Istanbul that has not been converted into a mosque.

 The Hagia Irene today operates as a museum and concert hall.
</item 9>

<item 10>
St. Mary's Church and Rectory (Iowa City, Iowa)
St. Mary's Catholic Church, also known as St. Mary of the Visitation Church, is a parish church of the Diocese of Davenport which is located at 228 E. Jefferson
St. in Iowa City, Iowa, United States.

 The church building and rectory are listed together on the National Register of Historic Places in 1980.

 They were both included as contributing properties in the Jefferson Street Historic District in 2004.

 The parish's first rectory, which is now a private home, is also listed on the National Register as St. Mary's Rectory.

 It is located a few blocks to the east of the present church location at 610 E. Jefferson St.
</item 10>
=== END ITEM id=68933738fe4b249b ===

=== ITEM id=787bc802cf3017a6 ===
Gold sentences and facts:
Sentence 0:  "Pinus" is the sole genus in the subfamily Pinoideae.
  Fact 0.0: Pinus is the sole genus in the subfamily Pinoideae.
Sentence 1:  The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35 unresolved species and many more synonyms.
  Fact 1.0: The Plant List was compiled by the Royal Botanic Gardens, Kew and the Missouri Botanical Garden.
  Fact 1.1: The Plant List accepts 126 species names of pines as current.
  Fact 1.2: The Plant List includes 35 unresolved species of pines.
  Fact 1.3: The Plant List includes many more synonyms.
Sentence 2:  It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
  Fact 2.0: The genus is sometimes considered to have only two species, B. monosperma and B. superba.
  Fact 2.1: The genus is sometimes expanded to include four or five species.

Retrieved items:
<item 1>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 1>

<item 2>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 2>

<item 3>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 3>

<item 4>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 4>

<item 5>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 5>

<item 6>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 6>

<item 7>
Pine
A pine is any conifer in the genus Pinus, , of the family Pinaceae.

 "Pinus" is the sole genus in the subfamily Pinoideae.

 The Plant List compiled by the Royal Botanic Gardens, Kew and Missouri Botanical Garden accepts 126 species names of pines as current, together with 35
unresolved species and many more synonyms.
</item 7>

<item 8>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 8>

<item 9>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 9>

<item 10>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 10>
=== END ITEM id=787bc802cf3017a6 ===

=== ITEM id=d0ac17b8bac5e09f ===
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
</item 1>

<item 2>
1999 British Grand Prix
The 1999 British Grand Prix was a Formula One motor race held on 11 July 1999 at the Silverstone Circuit near Silverstone, England.

 It was the eighth race of the 1999 Formula One season.

 The 60-lap race was won by McLaren driver David Coulthard after he started from third position.

 Eddie Irvine finished second for the Ferrari team and Williams driver Ralf Schumacher came in third.
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
Spyker F1
The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World
Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.

 The change to the Spyker name was accompanied by a switch in racing livery from the red and white previously used by Midland, to an orange and silver
scheme—already seen on the Spyker Spyder GT2-R—orange being the national colour and the auto racing colour of the Netherlands.

 At the end of the 2007 season the team was sold and renamed Force India.
</item 4>

<item 5>
Eddie Irvine
Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.

 He was a Formula One driver between 1993 and 2002, and runner-up in the 1999 World Drivers' Championship, driving for Scuderia Ferrari.
</item 5>

<item 6>
Jaguar R2
The Jaguar R2 was the car with which the Jaguar team competed in the Formula One season.

 It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.

 However, the Brazilian was replaced by Pedro de la Rosa after only four Grands Prix.
</item 6>

<item 7>
1983 Volvo International – Doubles
Sherwood Stewart and Ferdi Taygan were the defending champions but they competed with different partners that year, Stewart with Mark Edmondson and Taygan with
Cássio Motta.
</item 7>

<item 8>
1997 Argentine Grand Prix
The 1997 Argentine Grand Prix was a Formula One motor race held at Autódromo Oscar Alfredo Gálvez in Buenos Aires, Argentina on 13 April 1997.

 It was the third race of the 1997 Formula One season and the 600th World Championship event.

 The 72-lap race was won by Williams driver Jacques Villeneuve after he started from pole position.

 Eddie Irvine finished second for the Ferrari team and Jordan driver Ralf Schumacher came in third.
</item 8>

<item 9>
Graph query row:
person: Eddie Irvine
nationality: British
national_identity: None
</item 9>

<item 10>
Jaguar R3
The Jaguar R3 is a Formula One racing car with which Jaguar Racing competed in the 2002 Formula One season.

 The car was launched on January 4, 2002.

 It was driven by Eddie Irvine and Pedro de la Rosa, both retained from 2001.
</item 10>
=== END ITEM id=d0ac17b8bac5e09f ===

=== ITEM id=2623e6d9d3c758ac ===
Gold sentences and facts:
Sentence 0: Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.
  Fact 0.0: Karen Joy Fowler was born on February 7, 1950.
  Fact 0.1: Karen Joy Fowler is American.
  Fact 0.2: Karen Joy Fowler is an author of science fiction.
  Fact 0.3: Karen Joy Fowler is an author of fantasy.
  Fact 0.4: Karen Joy Fowler is an author of literary fiction.
Sentence 1: Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.
  Fact 1.0: Charles Bruce Chatwin was born on 13 May 1940.
  Fact 1.1: Charles Bruce Chatwin died on 18 January 1989.
  Fact 1.2: Charles Bruce Chatwin was English.
  Fact 1.3: Charles Bruce Chatwin was a travel writer.
  Fact 1.4: Charles Bruce Chatwin was a novelist.
  Fact 1.5: Charles Bruce Chatwin was a journalist.

Retrieved items:
<item 1>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 1>

<item 2>
Always (short story)
"Always" is a science fiction short story written by Karen Joy Fowler.

 Originally published in the April/May 2007 issue of "Asimov's Science Fiction", it won the 2007 Nebula Award for Best Short Story.

 It is collected in Fowler's "What I Didn't See".
</item 2>

<item 3>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 3>

<item 4>
Karen Joy Fowler
Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.

 Her work often centers on the nineteenth century, the lives of women, and alienation.
</item 4>

<item 5>
The entries profile writers, scholars, artists, publishers, and advocates. Highlights include computer scientist Nuria Oliver’s data-science leadership and
human-behavior research; Helen Keller’s childhood story “The Frost King”; Booker Prize winner Keri Hulme; Alice Walker’s Pulitzer-winning *The Color Purple* and
anti-FGM work; and Lucille Watahomigie’s Hualapai language revitalization. Others include novelists Sarra Manning and Kim Edwards, poet Tara Betts
</item 5>

<item 6>
Kate Atkinson’s 2015 *A God in Ruins*, companion to *Life After Life*, follows Teddy Todd in WWII and won the Costa Novel Award. Other works include John
Howard’s Central/Eastern European fiction; Conrad’s *Heart of Darkness*, Marlow’s Congo voyage and Kurtz obsession; Keri Hulme’s Booker-winning *The Bone
People*; Karen Joy Fowler’s PEN/Faulkner-winning, Booker-shortlisted *We Are All Completely Beside Ours
</item 6>

<item 7>
Bruce Chatwin
Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.

 His first book, "In Patagonia" (1977), established Chatwin as a travel writer, although he considered himself instead a storyteller, interested in bringing to
light unusual tales.

 He won the James Tait Black Memorial Prize for his novel "On the Black Hill" (1982) and his novel "Utz" (1988) was shortlisted for the Man Booker Prize.

 In 2008 "The Times" named Chatwin #46 on their list of "50 Greatest British Writers Since 1945."
</item 7>

<item 8>
Billy Kahora
Billy Kahora is a Kenyan writer and editor based in Nairobi.

 He was commended by the 2007 Caine Prize judges for his story "Treadmill Love".

 His stories "Urban Zoning" and "Gorilla’s Apprentice" were shortlisted for the prize in 2012 and 2014, respectively.

 He has written the non-fiction novella "The True Story of David Munyakei" the screenplay for "Soul Boy" and co-wrote "Nairobi Half Life".

 As Managing Editor of Kwani Trust, Kahora has edited seven issues of the "Kwani?

" journal.

 He is a contributing editor to the "Chimurenga Chronic".
</item 8>

<item 9>
The Jane Austen Book Club
The Jane Austen Book Club is a 2004 novel by American author Karen Joy Fowler.

 The story, which takes place near Sacramento, California, centers on a book club consisting of five women and one man who meet once a month to discuss Jane
Austen's six novels.

 It was a critical success and became a national bestseller.
</item 9>

<item 10>
We Are All Completely Beside Ourselves
We Are All Completely Beside Ourselves is a 2013 novel by the American writer Karen Joy Fowler.

 The novel won the 2014 PEN/Faulkner Award for Fiction<ref name="The Guardian - 3 April 2014 - Karen Joy Fowler wins PEN/Faulkner award for fiction"> </ref> and
was also short-listed for the 2014 Man Booker Prize.
</item 10>
=== END ITEM id=2623e6d9d3c758ac ===

=== ITEM id=58669e23df5d86c7 ===
Gold sentences and facts:
Sentence 0:  He has acted in a few Off Broadway productions and has made appearances on American television shows such as "", "Person of Interest" and "" and video games such as "Grand Theft Auto V", "Red Dead Redemption", "" and "Red Dead Revolver".
  Fact 0.0: This person has acted in a few Off Broadway productions.
  Fact 0.1: This person has made appearances on American television shows such as Person of Interest.
  Fact 0.2: This person has made appearances on an American television show whose title is blank in the sentence.
  Fact 0.3: This person has made appearances in video games such as Grand Theft Auto V.
  Fact 0.4: This person has made appearances in video games such as Red Dead Redemption.
  Fact 0.5: This person has made appearances in video games such as Red Dead Revolver.
  Fact 0.6: This person has made appearances in a video game whose title is blank in the sentence.
Sentence 1: Red Dead Redemption is a Western action-adventure video game developed by Rockstar San Diego and published by Rockstar Games.
  Fact 1.0: Red Dead Redemption is a Western action-adventure video game.
  Fact 1.1: Red Dead Redemption was developed by Rockstar San Diego.
  Fact 1.2: Red Dead Redemption was published by Rockstar Games.

Retrieved items:
<item 1>
Red Dead Redemption II
Red Dead Redemption II is an upcoming western action-adventure video game developed and published by Rockstar Games for release on PlayStation 4 and Xbox One in
Q2 2018.

 The game is a prequel to the 2010 title "Red Dead Redemption" and will be the third entry in the "Red Dead" series.

 It follows outlaw Arthur Morgan, a member of the Dutch van der Linde gang.
</item 1>

<item 2>
Tyler Bunch
Tyler T. Bunch (born February 10, 1970) is an American puppeteer, puppet designer, director, and actor.

 In addition to his work for the Jim Henson Company, Bunch has created and performed puppets for PBS, Nickelodeon and Disney's The Muppets Studio.

 He has acted in a few Off Broadway productions and has made appearances on American television shows such as "", "Person of Interest" and "" and video games
such as "Grand Theft Auto V", "Red Dead Redemption", "" and "Red Dead Revolver".

 Currently, he is directing the PBS children's television series "SeeMore's Playhouse".

 He is also currently scheduled to be one of the puppeteers for the musical stage adaptation of "Emmet Otter's Jug-Band Christmas".

 He resides in Stratford, Connecticut.
</item 2>

<item 3>
Development of Grand Theft Auto V
An approximately 1,000-person team developed "Grand Theft Auto V", an action-adventure video game, over several years.

 Rockstar Games released "Grand Theft Auto V" on 17 September 2013 for PlayStation 3 and Xbox 360, on 18 November 2014 for PlayStation 4 and Xbox One, and on 14
April 2015 for Microsoft Windows, as the fifteenth entry in the "Grand Theft Auto" series.

 Development began soon after "Grand Theft Auto IV"' s release and was led by Rockstar North's core 360-person team, who collaborated with several other
international Rockstar Games studios.

 The team considered the game a spiritual successor to many of their previous projects like "Red Dead Redemption" and "Max Payne 3".

 After its unexpected announcement in 2011, the game was fervently promoted with press showings, cinematic trailers, viral marketing strategies and special
editions. Its release date, though subject to several delays, was widely anticipated.
</item 3>

<item 4>
Red Dead Redemption
Red Dead Redemption is a Western action-adventure video game developed by Rockstar San Diego and published by Rockstar Games.

 It was released for PlayStation 3 and Xbox 360 consoles in May 2010.

 It is the second title in the "Red Dead" franchise, after 2004's "Red Dead Revolver".

 The game, set during the decline of the American Frontier in the year 1911, follows John Marston, a former outlaw whose wife and son are taken hostage by the
government in ransom for his services as a hired gun.

 Having no other choice, Marston sets out to bring the three members of his former gang to justice.
</item 4>

<item 5>
Controversies surrounding Grand Theft Auto V
"Grand Theft Auto V" is an open world, action-adventure video game developed by Rockstar North and published by Rockstar Games.

 Upon its 17 September 2013 release on PlayStation 3 and Xbox 360, the game generated controversies related to its violence and depiction of women.

 A mission that requires players to use torture equipment in a hostage interrogation polarised reviewers, who noted its political commentary but felt that the
torture sequence was in poor taste.

 The mission also received criticism from politicians and anti-torture charity groups.

 The game became subject to widespread online debate over its portrayal of women, particularly in the wake of backlash against GameSpot journalist Carolyn Petit
after she claimed the game was misogynystic in her review.

 After Petit's review webpage received more than 20,000 largely negative comments, many journalists defended her right to an opinion and lamented the gaming
community's defensiveness towards criticism.

 Television personality Karen Gravano and actress Lindsay Lohan both filed lawsuits against Rockstar in allegation that characters in the game were based on
their likenesses.
</item 5>

<item 6>
Grand Theft Auto: Vice City Stories
Grand Theft Auto: Vice City Stories is an open world action-adventure video game developed by Rockstar Leeds in conjunction with Rockstar North and published by
Rockstar Games.

 Released on 31 October 2006 for the PlayStation Portable and later for the PlayStation 2 on 5 March 2007.

 The game is the tenth instalment in the "Grand Theft Auto" series and the sixth and last game in the 3D universe.

 This was also the last game in the series to use well-known and Hollywood voice talent; "Grand Theft Auto IV" onward would instead use lesser-known and unknown
voice actors.

 It is a prequel to "" and successor to the previous PSP release, "".

 Vice City is a representation of Miami, Florida.
</item 6>

<item 7>
List of accolades received by Grand Theft Auto V
"Grand Theft Auto V" is an open world, action-adventure video game developed by Rockstar North and published by Rockstar Games.

 Players freely roam the fictional state of San Andreas (based on Southern California), composed of open countryside and the fictional city of Los Santos (based
on Los Angeles).

 The game was first announced on 25 October 2011, and was widely anticipated.

 At the 2012 Spike Video Game Awards (now the VGX), it was awarded Most Anticipated Game.

 It was released on 17 September 2013 for the PlayStation 3 and Xbox 360, on 18 November 2014 for the PlayStation 4 and Xbox One, and on 14 April 2015 for
Microsoft Windows.

 Review aggregator, Metacritic, assigned the game a normalised score of 97 out of 100, indicating "universal acclaim", based on 50 reviews for the PlayStation 3
version, 58 reviews for the Xbox 360 version, 66 reviews for the PlayStation 4 version and 14 reviews for the Xbox One version.
</item 7>

<item 8>
Red Dead Revolver
Red Dead Revolver is a 2004 western action-adventure third-person shooter video game developed by Rockstar San Diego, published by Rockstar Games and
distributed by Take-Two Interactive for the PlayStation 2 and Xbox.

 It is the first game in the "Red Dead" series.
</item 8>

<item 9>
List of Grand Theft Auto V characters
Rockstar North's 2013 open world action-adventure video game "Grand Theft Auto V", published by Rockstar Games, deals with a trio of criminals and their efforts
to commit heists while under pressure from a corrupt government agency.

 The game's use of three lead protagonists is a break from series tradition.

 Three days after its release, "Grand Theft Auto V" had earned more than US $1 billion in sales, making it the fastest selling video game in history.

 Additionally, "Grand Theft Auto V" was voted Game of the Year from several gaming publications.

 The game's characters were created by writers Dan Houser and Rupert Humphries.

 The use of multiple lead protagonists, as well as the cast in general, has received generally positive reviews from gaming magazines and websites.
</item 9>

<item 10>
Red Dead
Red Dead is a series of western action-adventure video games developed by Rockstar San Diego and published by Rockstar Games.

 The first game, titled "Red Dead Revolver", was originally released for the PlayStation 2 and Xbox in May 2004.

 The second game, "Red Dead Redemption", was released for the PlayStation 3 and Xbox 360 in May 2010 to wide critical acclaim, and is considered one of the
greatest games of all time.

 The third entry in the series, "Red Dead Redemption II", is scheduled for release in Q2 2018 for PlayStation 4 and Xbox One.
</item 10>
=== END ITEM id=58669e23df5d86c7 ===
