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

=== ITEM id=a26a1ebef48c9e29 ===
Gold sentences and facts:
Sentence 0: Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.
  Fact 0.0: Paola Suárez was born on 23 June 1976.
  Fact 0.1: Paola Suárez is a retired professional tennis player.
  Fact 0.2: Paola Suárez is from Argentina.
Sentence 1:  She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.
  Fact 1.0: Paola Suárez was one of the most prominent women's doubles players throughout the early and mid-2000s.
  Fact 1.1: Paola Suárez won eight Grand Slam titles.
  Fact 1.2: All eight of Paola Suárez's Grand Slam titles were won with Virginia Ruano Pascual.
  Fact 1.3: Paola Suárez held the No. 1 doubles ranking for 87 non-consecutive weeks.
Sentence 2:  He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.
  Fact 2.0: Paola Suárez reached a career high singles ranking of 274 in July 2011.
  Fact 2.1: Paola Suárez is much better known as a leading doubles player with a huge serve.

Retrieved items:
<item 1>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 1>

<item 2>
Paola Suárez
Paola Suárez (born 23 June 1976) is a retired professional tennis player from Argentina.

 She was one of the most prominent women's doubles players throughout the early and mid-2000s, winning eight Grand Slam titles, all of them with Virginia Ruano
Pascual, and holding the No. 1 doubles ranking for 87 non-consecutive weeks.

 She was also a singles semifinalist at the 2004 French Open.
</item 2>

<item 3>
At the 1976 Italian Open, Adriano Panatta and Mima Jaušovec won singles; Panatta survived 11 match points in round one. Jan Kodeš and Margaret Court won the
1970 French Open. Vitas Gerulaitis won Wimbledon doubles (1975), the 1977 Australian Open, Italian Opens (1977, 1979), and 1978 WCT Finals; he also won Brussels
(1982), Ocean City (1977
</item 3>

<item 4>
1999 Egger Tennis Festival – Doubles
The 1999 Egger Tennis Festival – Singles was the singles event of the twenty-ninth edition of the Egger Tennis Festival; a WTA Tier IV tournament and the second
most prestigious women's tennis tournament held in Austria.

 Laura Montalvo and Paola Suárez were the defending champions, but they didn't compete together this year.

 Montalvo played with Olga Lugina as the third seed, while Suárez teamed up with Virginia Ruano Pascual as the second seed.
</item 4>

<item 5>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 5>

<item 6>
1999 Copa Colsanitas – Singles
The 1999 Copa Colsanitas – Singles was the singles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Paola Suárez was the defending champion but lost in the semifinals to Christína Papadáki.
</item 6>

<item 7>
Clarisa Fernández
Clarisa Fernández (born 28 August 1981) is a retired Argentine tennis player who is best known for her semi-final appearance at the 2002 French Open, the first
Argentine since Sabatini in 1992 to reach that stage.

 Fernández was ranked 87th in the world at the time of her shocking result.

 She defeated, in order: (Q) Lubomira Bacheva, Jelena Kostanić, Kim Clijsters, (13) Elena Dementieva, Paola Suárez, before losing to second-seeded Venus
Williams in straight sets.
</item 7>

<item 8>
1999 Italian Open – Women's Doubles
The 1999 Italian Open – Doubles was the doubles event of the fifty-fifth edition of the tennis tournament played at Rome, Italy, the most prestigious tennis
tournament in Southern Europe.

 It was the fifth WTA Tier I tournament of the year, and part of the European claycourt season.

 Virginia Ruano Pascual and Paola Suárez were the defending champions, but lost to Martina Hingis and Anna Kournikova in the quarterfinals.
</item 8>

<item 9>
The material compiles tennis results and player biographies. Sherwood Stewart and Ferdi Taygan won the 1982 French Open doubles and finished runner-up at the
Volvo Masters and U.S. Pro Indoor; Stewart later partnered others. Jorge Lozano, a Mexican doubles No. 4, won nine doubles titles, including two French Open
mixed doubles crowns, and captained Mexico’s Davis Cup team. Brazilian Carlos Fernandes reached the 1961 French Open quarterfinals. Argentine Javier Fr
</item 9>

<item 10>
1999 Copa Colsanitas – Doubles
The 1999 Copa Colsanitas – Doubles was the doubles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Janette Husárová and Paola Suárez were the defending champions but only Suárez competed that year with Laura Montalvo.
</item 10>
=== END ITEM id=a26a1ebef48c9e29 ===

=== ITEM id=dbab0a7b2877e25c ===
Gold sentences and facts:
Sentence 0: Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.
  Fact 0.0: Victor Ronald Salva was born March 29, 1958.
  Fact 0.1: Victor Ronald Salva is an American film director.
  Fact 0.2: Victor Ronald Salva is an American screenwriter.
Sentence 1: Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.
  Fact 1.0: Emilio "El Indio" Fernández was born Emilio Fernández Romo.
  Fact 1.1: Emilio "El Indio" Fernández was born March 26, 1904.
  Fact 1.2: Emilio "El Indio" Fernández died August 6, 1986.
  Fact 1.3: Emilio "El Indio" Fernández was a Mexican film director.
  Fact 1.4: Emilio "El Indio" Fernández was a Mexican actor.
  Fact 1.5: Emilio "El Indio" Fernández was a Mexican screenwriter.

Retrieved items:
<item 1>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 1>

<item 2>
Filiberto Ojeda Ríos
Filiberto Ojeda Ríos (April 26, 1933 – September 23, 2005) was the commander-in-chief ("Responsable General") of the Boricua Popular Army ("Ejército Popular
Boricua, a.k.a., Los Macheteros").

 According to an unsigned article in the "Los Angeles Times", Los Macheteros was “a terrorist group seeking Puerto Rico's independence.”

 The group campaigned for, and supported, the independence of Puerto Rico from the United States.

 In 2001, FBI Director Louis J. Freeh claimed the group was linked to acts of terrorism, but Ronald Fernandez, scholar of Puerto Rican history, suggests such
labeling was an act of political convenience by the United States Government, intended to "shift the blame for any attacks on U.S. policy or personnel from us
to them".

 Ríos was also a founder of the FALN.

 In a 1983 New York Times article, Robert McFadden described the FALN as a Puerto Rican terrorist organization responsible for bombings during the 1970s and
early 1980s "in the name of Puerto Rican independence".
</item 2>

<item 3>
Columba Domínguez
Columba Domínguez Adalid (March 4, 1929 – August 13, 2014) was a Mexican actress, singer, and painter.

 Considered a crucial figure in the Golden Age of Mexican cinema, she was one of the muses of the film director Emilio Fernández, with whom she was romantically
linked for several years.

 She is remembered particularly for her performance in the film "Pueblerina" (1949), which is considered one of the jewels of the Mexican Cinema.
</item 3>

<item 4>
Antonio De Carlo
Antonio De Carlo, born (1967--) 4, 1967 (age 50 ) in Tijuana, Baja California, Mexico, is a Mexican actor also known as Antonio in the late 1980s and early
1990s.

 He won an Emmy Award in 2005-2006.

 He is president and founder of the "Fundación Cultura Sin Fronteras AC".

 After twelve years out of show business, in 2012 he returned to the soap operas performing "Magic Dragon" one of the main characters for "Miss XV", a
co-production from Televisa México and Nickelodeon directed and produced by Pedro Damian.

 He is also part of the cast of "Muchacha Italiana Viene a Casarse", directed by Pedro Damian (Televisa, México).
</item 4>

<item 5>
Valerio Fioravanti
Giuseppe Valerio "Giusva" Fioravanti (born 28 March 1958) is an Italian who, with Francesca Mambro, was a leading figure in a far right terrorist group "Nuclei
Armati Rivoluzionari" ("Armed Revolutionary Nuclei" or NAR).

 Fioravanti appeared in films and television from a young age, and in his early teens was the most famous child in Italy.

 He and Mambro were fugitives wanted for terrorist offences by their early twenties, they spent a further period on the run while wanted for the Bologna
bombing.

 Both were captured after gunfights with police.

 Although Fioravanti, Mambro and a third NAR mamber were convicted of the bombing, there were those who thought that a higher level of the conspiracy was never
uncovered.

 Fioravanti and Mambro admit carrying out terrorist murders, but have always denied any involvement in the Bologna station bomb.
</item 5>

<item 6>
Graph query row:
person1: Victor Salva
occupation1: ['film director', 'screenwriter']
person2: Emilio Fernández
occupation2: ['film director', 'actor', 'screenwriter']
shared_connection: None
relationship1: None
relationship2: None
</item 6>

<item 7>
Jaime Fernández (actor)
Jaime Fernández Reyes (December 6, 1937 – April 15, 2005) was a Mexican actor.

 Over his career, he won 3 Silver Ariel awards - the Mexican equivalent of the Oscar - including one for what is arguably his best-known role, playing Friday in
Luis Buñuel's "Robinson Crusoe".

 He appeared in over 200 films and served as the General Secretary of the Mexican actors' union for 11 years.

 His older brother was actor/director Emilio Fernández.
</item 7>

<item 8>
Emilio Fernández
Emilio "El Indio" Fernández (born Emilio Fernández Romo, ] ; March 26, 1904 – August 6, 1986) was a Mexican film director, actor and screenwriter.

 He was one of the most prolific film directors of the Golden Age of Mexican cinema in the 1940s and 1950s.

 He is best known for his work as director of the film "Maria Candelaria" (1944), which won the Palme d'Or award at the 1946 Cannes Film Festival.

 As an actor, he worked in numerous film productions in Mexico and in Hollywood.
</item 8>

<item 9>
Fernando Fernández (actor)
Fernando Fernández "El Crooner de México" (1916–1999) was a Mexican actor and singer.

 He was born on November 9, 1916, in Monterrey, Mexico, and died in Mexico City, Mexico, on November 24, 1999.

 Fernando was the son of Eloisa Reyes.

 He was brother of the famous film director Emilio Fernández and actor Jaime Fernández.

 He was married to singer Lupita Palomera, who died in 2008.

 Fernando Fernández was known as "the Crooner of Mexico."
</item 9>

<item 10>
Comedy in the Golden Age of Mexican cinema
Sadly, it was not as successful as many directors and film maker though it would be.

 Many failed to correctly synchronize the audio with the picture.

 Although to some extent it may have been unsuccessful, it is agreeable that it helped revolutionize film in Latin America.

 Mexico’s film industry takes its height post “El Cine Sonoro” to what is known as “The Golden Age”.

 Many film members consider this one of the most successful eras in the history of Mexican cinema.

 Many recognize the films of this era when they see black and white.

 It is also important to note that the films also portrayed social issues.

 Some of the most influential directors and actors are Emilio Fernández, Raul De Anda, Roberto B. Cervenna, Cantinflas, Vitola, Tin Tan, etc.

 The focus of this article will be on some of top grossing comedian actors of the time Cantinflas, Tin Tan, Clavillazo and others.
</item 10>
=== END ITEM id=dbab0a7b2877e25c ===

=== ITEM id=f7c6bad20fbba387 ===
Gold sentences and facts:
Sentence 0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
  Fact 0.0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book.
  Fact 0.1: Lawrence Weschler wrote Mr. Wilson's Cabinet of Wonder.
  Fact 0.2: Mr. Wilson's Cabinet of Wonder is primarily about the Museum of Jurassic Technology.
  Fact 0.3: Mr. Wilson's Cabinet of Wonder is more broadly about the history and role of museums.
Sentence 1: The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.
  Fact 1.0: The Museum of Jurassic Technology is a museum.
  Fact 1.1: The Museum of Jurassic Technology is located at 9341 Venice Boulevard.
  Fact 1.2: The Museum of Jurassic Technology is in the Palms district of Los Angeles, California.

Retrieved items:
<item 1>
Hagop Sandaldjian
Hagop Sandaldjian (1931–1990) was an Egyptian-born Armenian American musician and microminiature sculptor, best known for his tiny art pieces displayed at the
Museum of Jurassic Technology in Los Angeles, California.

 Sandaldjian's creations included a carving of Mount Ararat on a grain of rice; a crucifix in which a minute golden figure of Jesus hangs upon a cross made from
a bisected strand of Sandaldjian's own hair; and recreations of Disney figures (Snow White and the Seven Dwarfs or Mickey Mouse, for example) or historical
figures (such as Napoleon or Pope John Paul II) presented in the eye or on the tip of a needle.
</item 1>

<item 2>
Mr. Wilson's Cabinet of Wonder
Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily
about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
</item 2>

<item 3>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 3>

<item 4>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 4>

<item 5>
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
</item 5>

<item 6>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 6>

<item 7>
Dudley Zoo
Dudley Zoological Gardens is a 40 acre zoo located within the grounds of Dudley Castle in the town of Dudley, in the Black Country region of the West Midlands,
England.

 The Zoo opened to the public on 18 May 1937.

 Dudley Zoo is owned and operated by Dudley and West Midlands Zoological Society, founded in 1935 and a registered charity.
</item 7>

<item 8>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 8>

<item 9>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 9>

<item 10>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 10>
=== END ITEM id=f7c6bad20fbba387 ===

=== ITEM id=1a53f472aa2805d8 ===
Gold sentences and facts:
Sentence 0: Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.
  Fact 0.0: Batman is a 1989 American superhero film.
  Fact 0.1: Batman was directed by Tim Burton.
  Fact 0.2: Batman was produced by Jon Peters and Peter Guber.
  Fact 0.3: Batman is based on the DC Comics character of the same name.
Sentence 1:  He is Chairman and CEO of Mandalay Entertainment.
  Fact 1.0: The person referred to as 'he' is Chairman of Mandalay Entertainment.
  Fact 1.1: The person referred to as 'he' is CEO of Mandalay Entertainment.

Retrieved items:
<item 1>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 1>

<item 2>
Batman (1989 film)
Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.

 It is the first installment of Warner Bros.' initial "Batman" film series.

 The film stars Michael Keaton as Bruce Wayne/Batman and Jack Nicholson as The Joker, alongside Kim Basinger, Robert Wuhl, Pat Hingle, Billy Dee Williams,
Michael Gough and Jack Palance.

 The film takes place early in the title character's war on crime, and depicts a battle with his arch-nemesis the Joker.
</item 2>

<item 3>
Batman (album)
Batman is the eleventh studio album by American recording artist Prince and the soundtrack album to the 1989 film "Batman".

 It was released on June 20, 1989 by Warner Bros.

 Records.

 As a Warner Bros. stablemate, Prince's involvement in the soundtrack was designed to leverage the media company's contract-bound talent as well as fulfill the
artist's need for a commercial (if not critical) revival.

 The result was yet another multi-platinum successful cross-media enterprise by Warner Bros., in the vein of "Purple Rain".

 The album was No. 1 on the "Billboard" albums chart for six consecutive weeks.

 It has sold over eleven million copies worldwide.

 In 2016, film critic Matt Zoller Seitz praised Prince's songs and music videos for "Batman", more so than the film itself, stating that his songs "suggest a
goofy, perverse, sensuous, somewhat introverted Batman film that so far we've never gotten from anyone", and arguing that Prince's music videos "are more
psychologically perceptive than any of the Batman films."
</item 3>

<item 4>
Batman: Return of the Joker
Batman: Return of the Joker, known in Japan as Dynamite Batman (ダイナマイトバットマン , Dainamaito Battoman ) , is a 1991 platform video game, the follow-up to Sunsoft's
on the Nintendo Entertainment System.

 Unlike that game, which was based on the 1989 Batman film directed by Tim Burton, "Return of the Joker" is entirely self-contained and based more on the modern
comic book iteration of Batman.

 However, Batman mans the Batmobile from the 1989 film.

 A remake of "Return of the Joker", titled Batman: Revenge of the Joker, was released on the Sega Genesis by Ringler Studios in 1992.

 A Super NES version of "Revenge of the Joker" was in development, but never released.
</item 4>

<item 5>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 5>

<item 6>
Batman vs. Two-Face
Batman vs. Two-Face is an upcoming 2017 American animated direct-to-video superhero film produced by Warner Bros.

 Animation and distributed by Warner Bros.

 Home Entertainment.

 It is a sequel to "".

 It will premiere at the New York Comic Con on October 8, 2017 and will be followed by a digital release on October 10, 2017 and on DVD and Blu-ray on October
17, 2017.

 Based on the 1960s "Batman" TV series, the film stars Adam West (in his final role before his death), Burt Ward and Julie Newmar reprising their roles of
Batman, Robin and Catwoman from the series.

 The film will show a tribute to the late Adam West.

 This could also be the final time Warner Bros.

 Animation would make a 1960s Batman film.
</item 6>

<item 7>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 7>

<item 8>
Batman Forever (disambiguation)
Batman Forever is a 1995 film in the Batman film series.
</item 8>

<item 9>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
</item 9>

<item 10>
Peter Guber
Howard Peter Guber (born March 1, 1942) is an executive, entrepreneur, educator, and author.

 He is Chairman and CEO of Mandalay Entertainment.

 Guber's most recent films from Mandalay Entertainment include "The Kids Are All Right", "Soul Surfer", and "Bernie".

 He has also produced "Batman", "The Witches of Eastwick", and "Flashdance."

 Guber's films have earned over $3 billion worldwide and 50 Academy Award nominations.
</item 10>
=== END ITEM id=1a53f472aa2805d8 ===

=== ITEM id=52bc10fb5f907841 ===
Gold sentences and facts:
Sentence 0:  It has an area of 1208 km2 and had a population of 600,000 in 2003.
  Fact 0.0: Ezhou has an area of 1208 km2.
  Fact 0.1: Ezhou had a population of 600,000 in 2003.
Sentence 1: Ezhou () is a prefecture-level city in eastern Hubei Province, China.
  Fact 1.0: Ezhou is a prefecture-level city.
  Fact 1.1: Ezhou is in eastern Hubei Province.
  Fact 1.2: Ezhou is in China.
Sentence 2:  As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.
  Fact 2.0: As of the 2010 census, Ezhou had a population of 1,048,668.
  Fact 2.1: As of the 2010 census, 668,727 of Ezhou's residents lived in the core Echeng District.
Sentence 3:  The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
  Fact 3.0: The Ezhou-Huanggang built-up (or metro) area was home to 1,035,496 inhabitants.
  Fact 3.1: The inhabitants of the Ezhou-Huanggang built-up area came from the Echeng and Huangzhou districts of Huanggang.

Retrieved items:
<item 1>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 1>

<item 2>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 2>

<item 3>
Hulunbuir
Hulunbuir or Hulun Buir (Mongolian: ; , "Hūlúnbèi'ěr") is a region that is governed as a prefecture-level city in northeastern Inner Mongolia, in China.

 Its administrative center is located at Hailar District, its largest urban area.

 Major scenic features are the high steppes of the Hulun Buir grasslands, the Hulun and Buir lakes (the latter partially in Mongolia), and the Khingan range.

 Hulun Buir borders Russia to the north and west, Mongolia to the south and west, Heilongjiang province to the east and Hinggan League to the direct south.

 Hulunbuir is a linguistically diverse area: next to Mandarin Chinese, Mongolian dialects such as Khorchin and Buryat, the Mongolic language Dagur and some
Tungusic languages are spoken there.
</item 3>

<item 4>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 4>

<item 5>
Jiangyin dialect
Jiangyin dialect (江阴话) is a Northern Wu Chinese dialect spoken in the city of Jiangyin in Jiangsu province.

 Jiangyin dialect is a member of the Wu Chinese Taihu Wu family of dialects, which means the inhabitants speak a dialect similar to that of nearby Wuxi,
Changzhou, Suzhou, and Shanghai.

 Jiangyin dialect itself is of the Piling variety, related to the Changzhou dialect.

 Jiangyin dialect has the highest degree of mutual intelligibility with the dialects of the closest neighboring cities of Changzhou and Wuxi but also has a
fairly large degree of mutual intelligibility with the dialects of nearby Suzhou and Shanghai.

 As one travels south towards Wuxi away from the urban center of Jiangyin, Jiangyin dialect gradually becomes more and more closer sounding to the Wuxi dialect.
</item 5>

<item 6>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 6>

<item 7>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 7>

<item 8>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 8>

<item 9>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 9>

<item 10>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 10>
=== END ITEM id=52bc10fb5f907841 ===

=== ITEM id=15db15c6f108ff32 ===
Gold sentences and facts:
Sentence 0: The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.
  Fact 0.0: The Bishop of Ely is the ordinary of the Diocese of Ely.
  Fact 0.1: The Diocese of Ely is a Church of England diocese.
  Fact 0.2: The Diocese of Ely is in the Province of Canterbury.
Sentence 1:  Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
  Fact 1.0: Conway became Bishop of Ely in 2010.
  Fact 1.1: Conway was translated to Bishop of Ely from the Diocese of Salisbury.
  Fact 1.2: Conway was Bishop suffragan of Ramsbury before becoming Bishop of Ely.
  Fact 1.3: Conway was Bishop suffragan of Ramsbury in the Diocese of Salisbury.
Sentence 2: The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.
  Fact 2.0: The Diocese of Salisbury is a Church of England diocese.
  Fact 2.1: The Diocese of Salisbury is in the south of England.
  Fact 2.2: The Diocese of Salisbury is within the ecclesiastical Province of Canterbury.

Retrieved items:
<item 1>
Roman Catholic Ecclesiastical Province of Miami
The Ecclesiastical Province of Miami is a Catholic ecclesiastical province covering the U.S. state of Florida.

 Its metropolitan bishop is the Archbishop of Miami, head of the Archdiocese of Miami.

 The province additionally includes the suffragan dioceses of Orlando, Palm Beach, Pensacola-Tallahassee, St. Augustine, St. Petersburg, and Venice.
</item 1>

<item 2>
Ecclesiastical Province of Freiburg
The Ecclesiastical Province of Freiburg ("Kirchenprovinz Freiburg") or Upper Rhenish Ecclesiastical Province ("Oberrheinische Kirchenprovinz") is an
ecclesiastical province of the Roman Catholic Church in the Upper Rhine area of Germany, centring on Freiburg im Breisgau.

 It covers the Roman Catholic Archdiocese of Freiburg, the Roman Catholic Diocese of Mainz and the Roman Catholic Diocese of Rottenburg-Stuttgart, covering
large areas of Baden-Württemberg and Hesse and small parts of Rhineland-Palatinate.

 Its metropolitan bishop is the Archbishop of Freiburg - that Archdiocese and the Province were both set up in 1821 in the wake of the 1801 Concordat and the
1815 Congress of Vienna.
</item 2>

<item 3>
Ecclesiastical province
An ecclesiastical province is a general term for one of the basic forms of jurisdiction in Christian Churches with traditional hierarchical structure, including
Western Christianity and Eastern Christianity.

 In general, ecclesiastical province is consisted of several dioceses (or eparchies), one of them being the archdiocese (or archeparchy), headed by metropolitan
bishop or archbishop who has ecclesiastical jurisdiction over all other bishops of the province.
</item 3>

<item 4>
Ecclesiastical Province of Ontario
The Ecclesiastical Province of Ontario is one of the Anglican Church of Canada's four ecclesiastical provinces.

 It was established in 1912 out of six dioceses of the Ecclesiastical Province of Canada located in the civil Province of Ontario, and the Diocese of Moosonee
from the Ecclesiastical Province of Rupert's Land.
</item 4>

<item 5>
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 5>

<item 6>
Diocese of Salisbury
The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.

 The diocese covers most of Dorset (excepting the deaneries of Bournemouth and Christchurch, which fall within the Diocese of Winchester), and most of Wiltshire
(excepting a part in the north and Swindon).

 The diocese is led by the Bishop of Salisbury (Nick Holtam) and the diocesan synod.

 The bishop's seat is at Salisbury Cathedral.
</item 6>

<item 7>
Paul Burbridge
John Paul Burbridge was the Dean of Norwich in the latter part of the 20th century (1983-1995).

 Born on 21 May 1932, he was educated at The King's School, Canterbury and King's College, Cambridge, New College, Oxford, and Wells Theological College.

 After National Service with the Royal Artillery, he was ordained to a curacy at Eastbourne Parish Church in 1959.

 In 1962 he was appointed vicar choral and chamberlain at York Minster.

 He was appointed residentiary canon precentor at York Minster in 1966.

 In 1976 he was appointed Archdeacon of Richmond and canon residentiary at Ripon Cathedral, a post he held until his appointment as Dean of Norwich in 1983
(F.S.A).
</item 7>

<item 8>
Dean of Guildford
The Dean of Guildford is the head ("primus inter pares" – first among equals) and chair of the chapter of canons, the ruling body of Guildford Cathedral.

 The dean and chapter are based at the "Cathedral Church of the Holy Spirit" in Guildford.

 The cathedral is the mother church of the Diocese of Guildford and seat of the Bishop of Guildford.

 The current dean is Dianna Gwilliams, who was installed on 15 September 2013.
</item 8>

<item 9>
Roman Catholic Archdiocese of Cashel and Emly
The Archdiocese of Cashel and Emly (Irish: "Ard-Deoise Chaisil agus Imligh" ) is a Roman Catholic archdiocese in mid-western Ireland.

 The archdiocese is led by the Archbishop of Cashel and Emly, who serves as pastor of the mother church, the Cathedral of the Assumption and metropolitan of the
ecclesiastical province of Cashel and Emly.

 The Diocese of Cashel was established in 1111 by the Synod of Rathbreasail.

 The ecclesiastical province, which was roughly co-extensive with the secular province of Munster, was created in 1152 by the Synod of Kells.

 The cathedral church of the archdiocese is located in Thurles, County Tipperary.

 The incumbent archbishop is Kieran O'Reilly.
</item 9>

<item 10>
Bishop of Ely
The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.

 The diocese roughly covers the county of Cambridgeshire (with the exception of the Soke of Peterborough), together with a section of north-west Norfolk and has
its episcopal see in the City of Ely, Cambridgeshire, where the seat is located at the Cathedral Church of the Holy Trinity.

 The current bishop is Stephen Conway, who signs "+Stephen Elien:" (abbreviation of the Latin adjective "Eliensis", meaning "of Ely").

 The diocesan bishops now reside in Bishop's House, Ely, the former cathedral deanery.

 Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
</item 10>
=== END ITEM id=15db15c6f108ff32 ===

=== ITEM id=fb5fcfead0ef114e ===
Gold sentences and facts:
Sentence 0:  He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).
  Fact 0.0: The player played three seasons for the Philadelphia Eagles.
  Fact 0.1: The player's Philadelphia Eagles seasons are listed as 1940.
  Fact 0.2: The player played for the Pittsburgh Steelers.
  Fact 0.3: The player's Pittsburgh Steelers seasons are listed as 1941-1942.
Sentence 1:  He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his departure from the Eagles.
  Fact 1.0: The player was drafted by the Philadelphia Eagles.
  Fact 1.1: The player was drafted in the second round of the 2008 NFL Draft.
  Fact 1.2: The player played for the Washington Redskins for three seasons.
  Fact 1.3: The player played for the Washington Redskins after leaving the Eagles.

Retrieved items:
<item 1>
Graph query row:
person: Don Looney
shared_team: Philadelphia Eagles
connected_to: DeSean Jackson
</item 1>

<item 2>
Brooklyn Dodgers (NFL)
Later co-owner and partner Dan Topping (1912–1974), pulled the Tigers team out of the old NFL in 1946 and placed it in the newly established rival professional
league – the All-America Football Conference, which shortly lasted until 1949 until several stronger teams from the AAFC merged with and entered a reorganized
NFL in 1950.

 It lasted until 1970 with the NFL-AFL (third) merger following the establishment of the first "Super Bowl" inter-league national championship game three years
before with the old NFL champions playing the victors of the latest rival fourth incarnation of the American Football League IV, formed in 1960 (now the
American Football Conference (AFC).
</item 2>

<item 3>
Alec Gibson
Alec Raymond Gibson (born December 9, 1963 in Columbus, Ohio) is a former American football defensive end in the National Football League for the Washington
Redskins.

 Gibson played in three games for the Washington Redskins in 1987.

 The Washington Redskins went 3-0 during the strike.

 Defeating the St. Louis Cardinals, New York Giants, and Dallas Cowboys.

 Gibson caused Hall of Famer Tony Dorsett to fumble the football twice on Monday Night Football in Dallas the last replacement game played in 1987.

 He played college football at Ventura Junior College where he was an All-American defensive tackle and at the University of Illinois as a defensive end.

 In 1988, he played Arena football for the New York Knights.

 Gibson played 11 of 12 games before injuring his right knee (ACL) which put an end to his career.
</item 3>

<item 4>
Dallas Texans (NFL)
The Dallas Texans played in the National Football League (NFL) for one season, 1952, with a record of 1–11.

 The team is considered one of the worst teams in NFL history, both on (lowest franchise winning percentage) and off the field.

 The team was based first in Dallas, then Hershey, Pennsylvania, and Akron, Ohio, during its only season.

 The Texans were the last NFL team to fold.

 Many players on the 1952 roster went to the new Baltimore Colts franchise in 1953.

 The American Football League (AFL) had a 1960 charter member named the Dallas Texans (who later became the Kansas City Chiefs), but the AFL Texans have no
relationship with the earlier NFL team.
</item 4>

<item 5>
Jim O'Brien (American football)
Jim O'Brien (born February 2, 1947) is a former American football placekicker in the National Football League.

 He played for the Baltimore Colts from 1970 to 1972 and the Detroit Lions in 1973.

 He also played wide receiver, catching the bulk of his career passes during the 1972 season while still performing his kicking duties.

 His short career was less than stellar, posting a 55.6 percentage making 60 of 108 field goal attempts.

 His shining moment came in the closing moments of Super Bowl V in January 1971, where he kicked a 32-yard field goal with only five seconds remaining in the
game to break a tie and give the Colts the victory over the Dallas Cowboys 16–13.

 Before kicking the field goal, teammates saw that O'Brien was so nervous, he tried to take some of the artificial turf off the field to figure out the wind,
thinking the field was regular grass.

 Because of his singular moment kicking the Super Bowl-winning field goal, NFL Films named him the #9 "One-Hit Wonder" of all time.
</item 5>

<item 6>
1986 Tampa Bay Buccaneers season
The 1986 Tampa Bay Buccaneers season began with the team trying to improve on an 2–14 season.

 This was one of the worst seasons in franchise history.

 There is some sentiment that the 1986 team was even worse than the winless team of 1976, and the 473 points conceded was not beaten by any NFL team until the
2001 Indianapolis Colts gave up 486.

 The Buccaneers selected Bo Jackson with the top pick in the draft, but were unable to convince him to join the team.

 Three weeks after the draft, Jackson signed a three-year baseball contract with the Kansas City Royals.

 Despite holding four of the first forty selections in the draft, and the presence of a great influx of fresh talent from defunct USFL teams, the Buccaneers
were unable to find any impact players in either the draft or free agency.

 They entered the season with a roster nearly identical to the previous season's 2–14 team.
</item 6>

<item 7>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 7>

<item 8>
DeSean Jackson
DeSean William Jackson (born December 1, 1986) is an American football wide receiver for the Tampa Bay Buccaneers of the National Football League (NFL).

 He played college football for the University of California, Berkeley, where he was recognized as a consensus All-American.

 He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his
departure from the Eagles.

 Jackson has been selected to the Pro Bowl three times, and was the first player selected to the Pro Bowl at two different positions in the same year when he
was named to the 2010 Pro Bowl as a wide receiver and return specialist.
</item 8>

<item 9>
Verlon Biggs
Verlon Marion Biggs (March 16, 1943 – June 7, 1994) was an American football defensive end in the American Football League and National Football League.

 He played for the New York Jets (AFL) in Super Bowl III, but felt he didn't receive enough credit for the Jets' playoff win against the Oakland Raiders in the
AFL Championship Game that launched them into the Super Bowl.

 He played well but sulked until 1970, demanded more money, and wound up signing with Vince Lombardi's Washington Redskins of the NFL.

 Always a dominating defensive end, Verlon solidified George Allen's defense (Allen replaced Lombardi upon his death in July 1970) and led the Redskins into
Super Bowl VII.

 His nickname with the Redskins was "dirty Biggs" because of his extremely physical style of play.
</item 9>

<item 10>
Don Looney
John Don Looney (September 2, 1916 – April 5, 2015) was a professional American football end in the National Football League.

 He was born in Sulphur Springs, Texas.

 He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).

 He was the first receiver in NFL history to have over 100 yards receiving in each of his first two games, a feat which was not equaled until the 2008 NFL
season by another Eagles wide receiver, DeSean Jackson.

 At the time of his death, Looney was the second oldest living former NFL player.

 He was the father of NFL running back Joe Don Looney, who later died in a one-person motorcycle accident after his NFL career ended.
</item 10>
=== END ITEM id=fb5fcfead0ef114e ===

=== ITEM id=1cd45b071720c8a2 ===
Gold sentences and facts:
Sentence 0: Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.
  Fact 0.0: Samantha Cristoforetti was born 26 April 1977 in Milan.
  Fact 0.1: Samantha Cristoforetti is an Italian European Space Agency astronaut.
  Fact 0.2: Samantha Cristoforetti is an Italian Air Force pilot.
  Fact 0.3: Samantha Cristoforetti is an Italian engineer.
Sentence 1: Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force), and a former CNES spationaut.
  Fact 1.0: Jean-Loup Jacques Marie Chrétien was born 20 August 1938.
  Fact 1.1: Jean-Loup Jacques Marie Chrétien is French.
  Fact 1.2: Jean-Loup Jacques Marie Chrétien is retired.
  Fact 1.3: Jean-Loup Jacques Marie Chrétien holds the rank of Général de Brigade (brigadier general).
  Fact 1.4: Jean-Loup Jacques Marie Chrétien served as a general in the Armée de l'Air (French air force).
  Fact 1.5: Jean-Loup Jacques Marie Chrétien is a former CNES spationaut.

Retrieved items:
<item 1>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 1>

<item 2>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 2>

<item 3>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 3>

<item 4>
Samantha Cristoforetti
Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.

 She holds the record for the longest uninterrupted spaceflight of a European astronaut (199 days, 16 hours), and until June 2017 held the record for the
longest single space flight by a woman until this was broken by Peggy Whitson.

 She is also the first Italian woman in space.

 Samantha Cristoforetti is also known as the first person who brewed an espresso coffee in space.
</item 4>

<item 5>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 5>

<item 6>
Soyuz T-6
Soyuz T-6 was a manned spaceflight to Earth orbit to the Salyut 7 space station in 1982.

 Along with two Soviet cosmonauts, the crew included a Frenchman, Jean-Loup Chrétien.
</item 6>

<item 7>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 7>

<item 8>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 8>

<item 9>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 9>

<item 10>
Jean-Loup Puget
Jean-Loup Puget (born 7 March 1947) is a French astrophysicist.

 His current research interests lie in the Cosmic Microwave Background.

 Jean-Loup Puget and his collaborators reported the first identification of the Cosmic infrared background using COBE data.

 He is also, along with Alain Léger, credited with the origin of the hypothesis that the series of infrared lines observed in numerous astrophysical objects are
caused by emission from polycyclic aromatic hydrocarbons.

 He is currently principal investigator of the HFI module of the Planck space mission.
</item 10>
=== END ITEM id=1cd45b071720c8a2 ===

=== ITEM id=aa6c6768526d0cc4 ===
Gold sentences and facts:
Sentence 0: German Spitz is used to refer to both a breed of dog and category or type of dog.
  Fact 0.0: German Spitz is used to refer to a breed of dog.
  Fact 0.1: German Spitz is used to refer to a category or type of dog.
Sentence 1:  Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.
  Fact 1.0: Several modern breeds have been developed from the German Spitz.
  Fact 1.1: These modern breeds are either registered as separate breeds.
  Fact 1.2: These modern breeds are either registered as varieties of German Spitz.
Sentence 2:  The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that would later become known as the English Springer Spaniel.
  Fact 2.0: The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels.
  Fact 2.1: Attempts were made to use the term to specify a breed that would later become known as the English Springer Spaniel.

Retrieved items:
<item 1>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 1>

<item 2>
King Charles Spaniel
The King Charles Spaniel (also known as the English Toy Spaniel) is a small dog breed of the spaniel type.

 In 1903, the Kennel Club combined four separate toy spaniel breeds under this single title.

 The other varieties merged into this breed were the Blenheim, Ruby and Prince Charles Spaniels, each of which contributed one of the four colours available in
the breed.
</item 2>

<item 3>
Norfolk Spaniel
The Norfolk Spaniel or Shropshire Spaniel is an extinct breed of dog since the early 20th century.

 It was originally thought to have originated from the work of one of the Dukes of Norfolk, but this theory was disproven after being in doubt during the later
part of the 19th century.

 The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that
would later become known as the English Springer Spaniel.
</item 3>

<item 4>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 4>

<item 5>
English Springer Spaniel
The English Springer Spaniel is a breed of gun dog in the Spaniel family traditionally used for flushing and retrieving game.

 It is an affectionate, excitable breed with a typical lifespan of twelve to fourteen years.

 They are very similar to the Welsh Springer Spaniel and are descended from the Norfolk or Shropshire Spaniels of the mid-19th century; the breed has diverged
into separate show and working lines.

 The breed suffers from average health complaints.

 The show-bred version of the breed has been linked to "rage syndrome", although the disorder is very rare.

 It is closely related to the Welsh Springer Spaniel and very closely to the English Cocker Spaniel; less than a century ago, springers and cockers would come
from the same litter.

 The smaller "cockers" hunted woodcock while the larger littermates were used to flush, or "spring," game.

 In 1902, The Kennel Club recognized the English Springer Spaniel as a distinct breed.

 They are used as sniffer dogs on a widespread basis.

 The term "Springer" comes from the historic hunting role, where the dog would flush (spring) birds into the air.
</item 5>

<item 6>
Welsh Springer Spaniel
The Welsh Springer Spaniel is a breed of dog and a member of the spaniel family.

 Thought to be comparable to the old Land Spaniel, they are similar to the English Springer Spaniel and historically have been referred to as both the Welsh
Spaniel and the Welsh Cocker Spaniel.

 They were relatively unknown until a succession of victories in dog trials by the breed increased its popularity.

 Following recognition by The Kennel Club in 1902, the breed gained the modern name of Welsh Springer Spaniel.

 The breed's coat only comes in a single colour combination of white with red markings, usually in a piebald pattern.

 Loyal and affectionate, they can become very attached to family members and are wary of strangers.

 Health conditions are limited to those common among many breeds of dog, although they are affected more than average by hip dysplasia and some eye conditions.

 They are a working dog, bred for hunting, and while not as rare as some varieties of spaniel, they are rarer than the more widely known English Springer
Spaniel with which they are sometimes confused.
</item 6>

<item 7>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 7>

<item 8>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 8>

<item 9>
American Cocker Spaniel
The American Cocker Spaniel is a breed of sporting dog.

 It is a spaniel type dog that is closely related to the English Cocker Spaniel; the two breeds diverged during the 20th century due to differing breed
standards in America and the UK.

 In the United States, the breed is usually called the Cocker Spaniel, while elsewhere in the world, it is called the American Cocker Spaniel in order to
differentiate between it and its English cousin, which was already known as "Cocker Spaniel" before the American variety was created.

 The word "cocker" is commonly held to stem from their use to hunt woodcock in England, while "spaniel" is thought to be derived from the type's origins in
Spain.
</item 9>

<item 10>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 10>
=== END ITEM id=aa6c6768526d0cc4 ===

=== ITEM id=f796eb81ec62a9d4 ===
Gold sentences and facts:
Sentence 0: "Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the majority of their members quit and formed Faith No More.
  Fact 0.0: "Quiet in Heaven" / "Song of Liberty" is a double A-side single.
  Fact 0.1: "Quiet in Heaven" / "Song of Liberty" is the only official release by Faith No Man.
  Fact 0.2: Faith No Man is stylised as Faith•No Man•.
  Fact 0.3: The single was released before the majority of Faith No Man's members quit.
  Fact 0.4: After the majority of Faith No Man's members quit, they formed Faith No More.
Sentence 1:  Before settling on their current name in 1982, the band performed under the names Sharp Young Men and later Faith No Man.
  Fact 1.0: The band settled on its current name in 1982.
  Fact 1.1: Before settling on its current name in 1982, the band performed under the name Sharp Young Men.
  Fact 1.2: The band later performed under the name Faith No Man.

Retrieved items:
<item 1>
The Promised Land (Sagol 59 album)
A live show recording was released digitally in April 2017 as "The Promised Land at The Deanery, Brooklyn NY 3-23-2017".

 Live in Cafe Bialik, Tel Aviv, a soundboard concert recording, was released July 4th, 2017.
</item 1>

<item 2>
Kiss Me Quiet (song)
"Kiss Me Quiet" is a song recorded by Canadian country singer Jess Moskaluke for her second extended play of the same name (2015).

 It was written by Kelly Archer, Corey Crowder, and Emily Shackelton and was produced by Crowder.

 "Kiss Me Quiet" was released August 28, 2015 as the EP's lead single.

 The song was Moskaluke's first solo top 10 on the Canada Country chart.
</item 2>

<item 3>
Corcovado (song)
"Corcovado" (known in English as "Quiet Nights of Quiet Stars") is a bossa nova song written by Antônio Carlos Jobim in 1960.

 An English lyric was later written by Gene Lees.

 The Portuguese title refers to the Corcovado mountain in Rio de Janeiro.

 Andy Williams recorded the song with English lyrics, reaching #92 in the "Billboard" Hot 100 and #18 in the Hot Adult Contemporary Tracks chart in 1965.

 Also receiving air-play, contemporaneously with Andy Williams' recording of "Quiet Nights," was Kitty Kallen's version.

 Her album, titled "Quiet Nights," was released by 20th Century-Fox Records in 1964.
</item 3>

<item 4>
String Quartet No. 6 (Bartók)
Due to the difficulties of the war, communication between Bartók and Székely was difficult, and the quartet was not premiered until 20 January 1941, when the
Kolisch Quartet, to whom the work is dedicated, gave its premiere at the Town Hall in New York City.
</item 4>

<item 5>
Jefferson's Tree of Liberty
Jefferson's Tree of Liberty is the tenth album by Jefferson Starship, released on September 2, 2008.

 It is the band's first studio album since 1999's "Windows of Heaven".

 The new album includes cover songs from Irish, American, English, and Latin-American traditions.

 The title is a reference to Thomas Jefferson's quotation, "The tree of liberty must be refreshed from time to time with the blood of patriots and tyrants."

 The idea began in 2003 as "The Cuba Project", which was to include classic protest and folk songs recorded in Cuba.

 In 2008 the album was finally recorded but in California.

 About half of the songs planned for The Cuba Project were used on the final cut, with other songs coming from Jefferson Starship's previous repertoire and
another project band member Paul Kantner had planned called "On the Threshold of Fire."

 The promotional tour for the album began in late June with shows at Larkspur, California followed by tours in the US and Europe before the album's release, and
continued through December 2008 with a further tour in the US and a tour in Japan.
</item 5>

<item 6>
Quiet in Heaven/Song of Liberty
"Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the
majority of their members quit and formed Faith No More.

 The songs have not appeared in any of the band's compilation albums and have not been performed live since Mike Morris left the band.
</item 6>

<item 7>
- **Quiet Riot 10** (2014), the band’s twelfth studio album and first since reuniting in 2010, features Jizzy Pearl on six new studio tracks; its final four
tracks are live performances with Kevin DuBrow, who died in 2007.
- **Inner Secrets** (1978), Santana’s ninth album, shifted toward album-oriented rock. “Stormy” and “One Chain” were hits; “Well All Right” reached No.
</item 7>

<item 8>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 8>

<item 9>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 9>

<item 10>
Quiet Riot 10
Quiet Riot 10 (also alternatively known as just 10, or Quiet Riot Number 10) is the twelfth studio album by the heavy metal band Quiet Riot, which was released
on June 27, 2014.

 It is their first studio album since 1988's "QR" not to feature longtime and founding vocalist Kevin DuBrow in any newly recorded material, due to his death in
November 2007.

 It is also the band's first album since reuniting in 2010.

 Although a studio album, the final four tracks on "Quiet Riot 10" are live performances taken from some of the band's final shows with DuBrow in 2007.

 Love/Hate vocalist Jizzy Pearl joined the band in November 2013 and performs lead vocals on the six studio tracks.
</item 10>
=== END ITEM id=f796eb81ec62a9d4 ===

=== ITEM id=65788f49907ec8c1 ===
Gold sentences and facts:
Sentence 0:  It is found in Yemen and India.
  Fact 0.0: The subject of this sentence is found in Yemen.
  Fact 0.1: The subject of this sentence is found in India.
Sentence 1:  Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast.
  Fact 1.0: Yemen's capital has been temporarily relocated to the port city of Aden.
  Fact 1.1: Aden is on the southern coast.
  Fact 1.2: The relocation is stated to be because of the content of the previous sentence.

Retrieved items:
<item 1>
Aulacodes traversalis
Aulacodes traversalis is a moth in the Crambidae family.

 It was described by Harrison Gray Dyar Jr. in 1914.

 It is found in Panama.
</item 1>

<item 2>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 2>

<item 3>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 3>

<item 4>
Ornithodoros savignyi
Because of its habit of feeding and dropping from its host, adult dispersal is limited, whereas larvae may remain attached to their hosts for several days.

 During its life cycle it will feed on multiple hosts between moults.
</item 4>

<item 5>
Aulacodes templalis
Aulacodes templalis is a moth in the Crambidae family.

 It was described by Schaus in 1906.

 It is found in Brazil.
</item 5>

<item 6>
The entries describe diverse plants and one crab genus. Mascarene Dictyosperma album, the wind-shedding “hurricane palm,” is widely cultivated but nearly
extinct in its native range. Canary Islands Adenocarpus viscosus grows above 1,800 m. Dregea vines occur in Africa and southern Asia; Duranta shrubs and trees
are ornamental, with D. erecta invasive in several regions. Tropical Dieffenbachia are popular houseplants
</item 6>

<item 7>
The entries cover diverse taxa and concepts: butterflies (Mormon metalmark, tiger leafwing, Indian/Malayan leafwing, and Ionolyce blues), moths (three Aulacodes
species and the pale western cutworm), birds (ground jays, honeyguides, black terns), and other organisms. They note habitats, ranges, diets, camouflage, and
ecological roles. Also described are sand tampan ticks, midge orchids, Cretaceous herbivorous
</item 7>

<item 8>
Agrotis orthogonia
The Pale Western Cutworm ("Agrotis orthogonia") is a moth of the Noctuidae family.

 It is found in North America, more specifically dry, semi-desert areas of western North America from southern Canada to California, ranging eastward nearly to
the eastern edge of the Great Plains.
</item 8>

<item 9>
Al Khor (city)
Al Khor is a coastal city in northern Qatar, located 50 kilometres north of the capital, Doha.

 It is the capital city of the municipality of Al Khor.

 The name of the city means creek in Arabic as the town is located on a creek.

 Al Khor is home to many employees of the oil industry due to its proximity to Qatar's northern oil and natural gas fields, and due to its proximity to the Ras
Laffan Industrial City.
</item 9>

<item 10>
Amka
Amka (Hebrew: עַמְקָה‎ ), also known in Arabic as Amqa (Arabic: عمقا‎ ‎ ), is a moshav in the Matte Asher Regional Council of Israel's Northern District, near
Acre.

 The location of the moshav roughly corresponds the former Arab village, depopulated during the 1948 Arab–Israeli War.

 Yemenite Jews, who arrived from the southern Arab country of Yemen, founded the village's successor Amka in 1949.

 In 2016 its population was 706 .
</item 10>
=== END ITEM id=65788f49907ec8c1 ===
