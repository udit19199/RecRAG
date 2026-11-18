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

=== ITEM id=093a220fc6abdd4b ===
Gold sentences and facts:
Sentence 0: Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.
  Fact 0.0: Alice Malsenior Walker was born on February 9, 1944.
  Fact 0.1: Alice Malsenior Walker is an American novelist.
  Fact 0.2: Alice Malsenior Walker is an American short story writer.
  Fact 0.3: Alice Malsenior Walker is an American poet.
  Fact 0.4: Alice Malsenior Walker is an American activist.
Sentence 1: Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.
  Fact 1.0: Michael David Herr was born on April 13, 1940.
  Fact 1.1: Michael David Herr died on June 23, 2016.
  Fact 1.2: Michael David Herr was an American writer.
  Fact 1.3: Michael David Herr was an American war correspondent.
  Fact 1.4: Michael David Herr is known as the author of the memoir 'Dispatches', published in 1977.
  Fact 1.5: 'Dispatches' is a memoir of Michael David Herr's time as a correspondent for 'Esquire magazine' from 1967 to 1969.
  Fact 1.6: Michael David Herr was a correspondent during the Vietnam War.

Retrieved items:
<item 1>
Graph query row:
full_name: Michael David Herr
middle_name: David
</item 1>

<item 2>
Alice R. Ballard
Alice R. Ballard.

 (born June 16, 1945, Florence, South Carolina) is an American ceramicist based in Greenville, South Carolina.

 Much of her work is characterized by the organic earthenware forms of closed containers, pinch pots, platters, pods, teapots, totems, small work, vessels and a
series she refers to as her white work.

 Much of her work is finished with terra sigillata.

 Alice considers her art to be “a reflection of [her] relationship with natural forms.

 It is often the metamorphosis of Nature's forms, as they change from season to season, that attracts her to that universal world in which differing life forms
share similar qualities."

 Her aesthetic heavily references the mother/child/germination metaphor and also explores the more evocative realm of wonder and awe.
</item 2>

<item 3>
Graph query row:
full_name: Alice Malsenior Walker
middle_name: Malsenior
</item 3>

<item 4>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 4>

<item 5>
Haniel Long
Haniel Clark Long (March 9, 1888 – October 17, 1956) was an American poet, novelist, publisher and academic.

 He is best known for his novella, "Interlinear to Cabeza de Vaca" (1936), a fictionalized account of the true story of a Spanish conquistador in 16th century
North America.
</item 5>

<item 6>
Tobe Levin
Alice Walker expressed appreciation for the text that shows worldwide solidarity with the novelist's literary abolition efforts in the early nineties.

 Levin has also teamed up with Maria Kiminta and photographer Britta Radike to publish a memoir and sourcebook, "Kiminta.

 A Maasai's Fight against Female Genital Mutilation."
</item 6>

<item 7>
Warrior Marks
Warrior Marks: Female Genital Mutilation and the Sexual Blinding of Women is a 1993 book by Alice Walker with Pratibha Parmar, who made an award-winning
documentary of the same name.

 Following on from her 1992 novel "Possessing the Secret of Joy", Walker undertakes a journey to parts of Africa where clitoridectomy is still practised.

 "Warrior Marks" is a harrowing work as Walker interviews women who have had the operation done and finally interviews a woman—circumcised herself—who performs
the operation.
</item 7>

<item 8>
Alice Walker: Beauty in Truth
Alice Walker: Beauty in Truth is a documentary film directed by Pratibha Parmar, made by Kali Films production company.

 The film follows the life of the Pulitzer Prize-winning author, poet and activist Alice Walker.

 Shooting began in May 2011.

 It was aired on the BBC on Monday July 8, 2013, and on PBS on February 7, 2014.
</item 8>

<item 9>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 9>

<item 10>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 10>
=== END ITEM id=093a220fc6abdd4b ===

=== ITEM id=fd627e2afcd5dd13 ===
Gold sentences and facts:
Sentence 0: Gasland is a 2010 American documentary written and directed by Josh Fox.
  Fact 0.0: Gasland is a documentary.
  Fact 0.1: Gasland is from 2010.
  Fact 0.2: Gasland is American.
  Fact 0.3: Gasland was written by Josh Fox.
  Fact 0.4: Gasland was directed by Josh Fox.
Sentence 1: To Shoot an Elephant is a 2009 documentary film about the 2008-2009 Gaza War directed by Alberto Arce and Mohammad Rujailahk.
  Fact 1.0: To Shoot an Elephant is a 2009 documentary film.
  Fact 1.1: To Shoot an Elephant is about the 2008-2009 Gaza War.
  Fact 1.2: To Shoot an Elephant was directed by Alberto Arce.
  Fact 1.3: To Shoot an Elephant was directed by Mohammad Rujailahk.

Retrieved items:
<item 1>
The Last Lions
The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.

 It was shot at the Okavango Delta in Botswana.

 The film premiered at the Palm Springs International Film Festival in January 2011 and was released in select theaters on February 18, 2011.

 The film follows in the tradition of other National Geographic big cat films, such as "India: Land of the Tiger" and "Eye of the Leopard".
</item 1>

<item 2>
Elephant (album)
Elephant is the fourth album by the American alternative rock duo The White Stripes.

 Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the
Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album
charts.
</item 2>

<item 3>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 3>

<item 4>
Gasland
Gasland is a 2010 American documentary written and directed by Josh Fox.

 Nominated for an Academy Award for Best Documentary in 2011, the film focuses on communities in the United States affected by natural gas drilling and,
specifically, a method of horizontal drilling into shale formations known as hydraulic fracturing.

 The film was a key mobilizer for the anti-fracking movement, and "brought the term 'hydraulic fracturing' into the nation's living rooms" according to "The New
York Times".

 The film was nominated for the Academy Award for Best Documentary Feature.
</item 4>

<item 5>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 5>

<item 6>
To Shoot an Elephant
To Shoot an Elephant is a 2009 documentary film about the 2008-2009 Gaza War directed by Alberto Arce and Mohammad Rujailahk.
</item 6>

<item 7>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 7>

<item 8>
Hunting Elephants
Hunting Elephants (Hebrew: לצוד פילים‎ ‎ , "Latzud Pilim", "lit."

 To Hunt Elephants) is a 2013 Israeli crime comedy film directed by Reshef Levi.

 It stars, among others, Sasson Gabai, Moni Moshonov and Patrick Stewart.

 It was released to theaters in Israel on July 4, 2013.
</item 8>

<item 9>
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
</item 9>

<item 10>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 10>
=== END ITEM id=fd627e2afcd5dd13 ===

=== ITEM id=348c5ab2ec5b93b2 ===
Gold sentences and facts:
Sentence 0: Robert Harry Philibosian (born 1940) is an American politician.
  Fact 0.0: Robert Harry Philibosian was born in 1940.
  Fact 0.1: Robert Harry Philibosian is an American politician.
Sentence 1: David R. Ignatius (May 26, 1950), is an American journalist and novelist.
  Fact 1.0: David R. Ignatius was born on May 26, 1950.
  Fact 1.1: David R. Ignatius is an American journalist.
  Fact 1.2: David R. Ignatius is an American novelist.

Retrieved items:
<item 1>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 1>

<item 2>
David I. Walsh
David Ignatius Walsh (November 11, 1872June 11, 1947) was a United States politician from Massachusetts.

 A member of the Democratic Party, he served as the 46th Governor of Massachusetts before serving several terms in the United States Senate.
</item 2>

<item 3>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 3>

<item 4>
Mark Krikorian
Mark Krikorian has been the executive director of the Center for Immigration Studies, a think-tank in Washington, D. C., since 1995.

 The Center describes itself as an "independent, non-partisan research organization" in Washington, D. C., that examines and critiques the impact of immigration
on the United States.

 Animated by a "pro-immigrant, low-immigration vision which seeks fewer immigrants, but a warmer welcome for those admitted", the Center was established in 1985
to provide immigration research.

 Krikorian is a regular contributor to the conservative publication "National Review", and is a regular participant at "National Review Online's" "The Corner."
</item 4>

<item 5>
Robert Philibosian
Robert Harry Philibosian (born 1940) is an American politician.

 He served as Los Angeles County District Attorney from 1982 to 1984.

 He received his B.A. in history from Stanford University and his law degree from Southwestern Law School, and was admitted to the California State Bar in 1968.

 He is now Of counsel at the law firm of Sheppard, Mullin, Richter & Hampton.
</item 5>

<item 6>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 6>

<item 7>
Richard Ziser
Richard Ziser is a Nevada Real Estate Investor, Socially Conservative Political activist and U.S. Republican Politician.

 He was born June 7, 1953, in Pomona, CA., and has resided in Las Vegas Nevada since 1991.

 He graduated from California State Polytechnic University, Pomona (Cal Poly Pomona) with a BS in Industrial Engineering, 1976; then subsequently from Simon
Greenleaf University in Santa Ana, Ca.

 (now a campus of Trinity International University, with an MA in Christian Apologetics in 1989.
</item 7>

<item 8>
David Smick
David M. Smick is a macroeconomic adviser to a select group of prominent global investment funds and a non-fiction author.

 He is the chairman and CEO of Johnson Smick International, a financial advisory firm in Washington, D.C. where he is in partnership with former Federal Reserve
Vice Chairman Manuel H. Johnson.

 Smick wrote an acclaimed book "The World Is Curved" on the financial perils of globalization, in response to "The World Is Flat" bestseller by Thomas Friedman.

 The book received positive reviews by the "USA Today" and "Forbes", was called "astonishingly prescient" by David Brooks and "clairvoyant" by David Ignatius.
</item 8>

<item 9>
Andrew Benjamin
Andrew Benjamin (born 1952, Australia) is an Australian philosopher.

 Currently he holds two academic posts: Distinguished Professor of Philosophy and the Humanities at Kingston University, London and Professor of Philosophy and
Jewish Thought in the Department of Philosophy and the Australian Centre for Jewish Civilization at Monash University, Melbourne, Australia.

 Benjamin first came to critical attention with his writings in continental philosophy, writing articles and editing books on the thinking of Martin Heidegger,
Theodor Adorno, Walter Benjamin, Julia Kristeva and Jean-François Lyotard.

 Benjamin has become involved in the field of architecture, to the extent that he has also taught in various schools of architecture in UK, US and Australia.
</item 9>

<item 10>
Phil Driscoll
Phil Driscoll (born November 9, 1947) is a trumpeter, singer, composer, and producer.

 He performs in varying music genres and styles which include rock and roll, rhythm and blues, and patriotic music, and is best known for his work in Christian
music and his longterm Christian ministry.

 In 1985, Driscoll won the Grammy Award for Best Gospel Performance – Duo/Group for a duet with Debby Boone on "Keep the Flame Burning", and he has been
nominated for three additional Grammys, two for Best Gospel Performance – Male and one for Best Gospel/Pop Album.

 He has also won three Dove Awards for his music, and the 1999 Christian Country Music Association Award for Best Musician.

 In 2006, Driscoll was found guilty on 2 counts of tax evasion and one count of conspiracy, and was sentenced to serve one year in Federal prison, beginning on
March 14, 2007.
</item 10>
=== END ITEM id=348c5ab2ec5b93b2 ===

=== ITEM id=04c3ac7fe200e2cc ===
Gold sentences and facts:
Sentence 0: Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.
  Fact 0.0: Pagliacci is an Italian opera.
  Fact 0.1: Pagliacci has a prologue and two acts.
  Fact 0.2: Pagliacci has music and libretto by Ruggero Leoncavallo.
  Fact 0.3: The literal translation of Pagliacci is 'Clowns'.
Sentence 1: Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.
  Fact 1.0: Suor Angelica (Sister Angelica) is an opera.
  Fact 1.1: Suor Angelica is in one act.
  Fact 1.2: Suor Angelica is by Giacomo Puccini.
  Fact 1.3: Suor Angelica has an original Italian libretto by Giovacchino Forzano.

Retrieved items:
<item 1>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 1>

<item 2>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 2>

<item 3>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 3>

<item 4>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 4>

<item 5>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 5>

<item 6>
Senza sangue
Senza sangue (Without blood) an opera in one act by Péter Eötvös with a libretto by Mari Mezei.

 It is based on the second part of the novel of the same name by Alessandro Baricco and is intended to be performed jointly with Béla Bartók's one-act
"Bluebeard's Castle".

 The concert première was given by the Cologne Philharmonic Orchestra on 1 May 2015.

 It was premièred in full on 15 May 2016 at the Festival d'Avignon.

 It is his tenth opera.

 Anne Sofie von Otter, who gave the New York première with the New York Philharmonic, said: "“The piece is not at all easy for the two singers... [It’s] hard to
find the pitch; you have to work it into the voice, unless of course you happen to have perfect pitch, which I don’t."
</item 6>

<item 7>
Suor Angelica
Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.

 It is the second opera of the trio of operas known as "Il trittico" ("Triptych").

 It received its world premiere at the Metropolitan Opera on December 14, 1918.
</item 7>

<item 8>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 8>

<item 9>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 9>

<item 10>
Rita Fornia
Rita Fornia (17 July 1878 – 27 October 1922) was an American opera singer.

 She began her career in Europe at the beginning of the 20th century singing coloratura soprano roles.

 Early on in her career her voice darkened and dropped slightly causing her to focus more within the mezzo-soprano repertoire while still singing some soprano
roles.

 She joined the Metropolitan Opera company in 1907 where she performed regularly in mostly supporting roles until her retirement in 1922.

 She is best remembered today for portraying the role of the Abbess in the original production of Puccini's "Suor Angelica" in 1918.
</item 10>
=== END ITEM id=04c3ac7fe200e2cc ===

=== ITEM id=b736cab3fc16f05b ===
Gold sentences and facts:
Sentence 0:  It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
  Fact 0.0: The series was created by Jay Lender.
  Fact 0.1: The series was created by Micah Wright.
  Fact 0.2: Jay Lender was a former writer for The Angry Beavers.
  Fact 0.3: Jay Lender was a former writer for SpongeBob SquarePants.
  Fact 0.4: Jay Lender was a former writer for Phineas and Ferb.
  Fact 0.5: Micah Wright was a former writer for The Angry Beavers.
  Fact 0.6: Micah Wright was a former writer for SpongeBob SquarePants.
  Fact 0.7: Micah Wright was a former writer for Phineas and Ferb.
Sentence 1:  The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.
  Fact 1.0: The series chronicles the adventures and endeavors of the title character and his various friends.
  Fact 1.1: The series is set in the fictional underwater city of Bikini Bottom.

Retrieved items:
<item 1>
List of SpongeBob SquarePants episodes
"SpongeBob SquarePants" is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series is set in the fictional underwater city of Bikini Bottom, and centers on the adventures and endeavors of SpongeBob SquarePants, an over-optimistic
sea sponge that annoys other characters.

 Many of the ideas for the show originated in an unpublished, educational comic book titled "The Intertidal Zone", which Hillenburg created in the mid-1980s.

 He began developing "SpongeBob SquarePants" into a television series in 1996 after the cancellation of "Rocko's Modern Life", another Nickelodeon television
series which Hillenburg previously directed.
</item 1>

<item 2>
Robocalypse
Robocalypse is a real-time strategy video game developed and published by Vogster Entertainment for the Nintendo DS handheld video game console.

 It has only been released in North America.

 It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
</item 2>

<item 3>
Right Now Kapow
Right Now Kapow is an American animated sketch comedy television series produced by Warner Bros.

 Animation for Disney XD.

 It is the first collaboration between Warner Bros.

 Animation and Disney.

 The series premiered on September 19, 2016 and ended on May 31, 2017.

 The series was created by Justin Becker and Marly Halpern-Graser, who previously worked on the Cartoon Network series "Mad".

 Becker also worked on Adult Swim infomercials, and Halpern-Graser also worked on Cartoon Network's "DC Nation".

 The series follows Dog, Candy, Ice Cream, Diamond, Plant, and Moon going on new adventures everyday...and find themselves in mischief.

 Every episode of the series has 3 main parts in every 11-minute segment and other random shorts.

 The series has a style of humor similar to Cartoon Network's previous series, "Mad".
</item 3>

<item 4>
Rock Bottom (SpongeBob SquarePants)
"Rock Bottom" is the 34th episode of the first season of the American animated television series "SpongeBob SquarePants".

 It originally was produced in 1999 and aired on Nickelodeon in the United States on March 15, 2000.

 The series follows the adventures of the title character in the underwater city of Bikini Bottom.

 In the episode, SpongeBob becomes stranded in a abyssopelagic zone called Rock Bottom.
</item 4>

<item 5>
SpongeBob SquarePants (season 3)
The third season of the American animated television series "SpongeBob SquarePants", created by Stephen Hillenburg, aired on Nickelodeon from October 5, 2001 to
October 11, 2004, and consists of 20 episodes.

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, who also acted as the showrunner.

 Hillenburg halted production on the show to work on the 2004 film adaptation of the series, "The SpongeBob SquarePants Movie".

 After production on the film, Hillenburg resigned from the show as its showrunner, and appointed staff writer Paul Tibbitt to overtake the position.

 Season 3 was originally set to end the series after the release of the film, but the success prevented the series from ending, leading to a fourth season.
</item 5>

<item 6>
Band Geeks
"Band Geeks" is the second part of the 15th episode of the second season, and the second half of the 35th episode overall, of the American animated television
program "SpongeBob SquarePants".

 It originally aired on Nickelodeon in the United States on September 7, 2001.

 It was written by C. H. Greenblatt, Aaron Springer, and Merriwether Williams, and the animation was directed by Frank Weiss.

 Springer served as storyboard director, and Greenblatt served as storyboard artist.

 The song "Sweet Victory" by David Glen Eisley was featured in the episode and was later released on the album "" in 2005.
</item 6>

<item 7>
SpongeBob SquarePants: Battle for Bikini Bottom
SpongeBob SquarePants: Battle for Bikini Bottom is a 2003 video game based on the animated series of the same name, developed by Heavy Iron Studios, AWE Games,
and Vicarious Visions, and published by THQ.

 The game was released for the PlayStation 2, Game Boy Advance (GBA), Xbox, and Nintendo GameCube consoles as well as a port for Microsoft Windows.

 The Microsoft Windows version of the game is a point-and-click adventure game, whereas the 6th generation platforms received a 3D platformer.

 The Game Boy Advance version received a 2D platformer.

 All versions of the game feature an original storyline, in which the player attempts to defend Bikini Bottom from an invasion of robots created by Plankton
with a machine called the Duplicatotron 3000, playing as SpongeBob in all versions, as well as Patrick and Sandy in the console versions.

 The game was released on October 31, 2003, in North America and in Europe on November 28, 2003.
</item 7>

<item 8>
SpongeBob SquarePants (season 4)
The fourth season of the American animated television series "SpongeBob SquarePants", created by former marine biologist and animator Stephen Hillenburg, aired
on Nickelodeon from May 6, 2005 to July 24, 2007, and contained 20 episodes, beginning with the episodes "Fear of a Krabby Patty" and "Shell of a Man".

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, while writer Paul Tibbitt acted as the supervising/co-executive producer and showrunner.

 The show underwent a hiatus on television as Hillenburg halted the production in 2002 to work on the film adaptation of the series, "The SpongeBob SquarePants
Movie".

 Once the film was finalized and the previous season had completed broadcast on television, Hillenburg wanted to end the show, but the success of the series led
to more episodes, so Tibbitt took over Hillenburg's position as showrunner and began working on a fourth season for broadcast in 2005.
</item 8>

<item 9>
SpongeBob SquarePants
SpongeBob SquarePants is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The series' popularity has made it a media franchise, as well as the highest rated series to ever air on Nickelodeon, and the most distributed property of MTV
Networks.

 As of 2015, the media franchise has generated $12 billion in merchandising revenue for Nickelodeon.
</item 9>

<item 10>
Legends of Bikini Bottom
Legends of Bikini Bottom is an anthology series of six episodes in the American animated television series "SpongeBob SquarePants", as part of its seventh
season.

 As the name suggests, the episodes have plots involving things like monsters, magic and mythical creatures.

 The first episode called "Trenchbillies" first premiered on the online social networking service Facebook on January 27, 2011 before airing on Nickelodeon on
January 29, 2011.

 The four other episodes titled "Sponge-Cano!"

, "The Main Drain", "The Monster Who Came to Bikini Bottom" and "Welcome to the Bikini Bottom Triangle" premiered on Nickelodeon as part of an hour-long special
on January 28, 2011.

 The decision to air the series online was aimed at attracting "the young and the restless hooked to the internet and the social media."
</item 10>
=== END ITEM id=b736cab3fc16f05b ===

=== ITEM id=266c467dfddc5a27 ===
Gold sentences and facts:
Sentence 0:  It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke Bader and Linda Riley.
  Fact 0.0: The publication was launched in March 1994.
  Fact 0.1: The publication was launched by Millivres Ltd.
  Fact 0.2: The publication was launched under the editorship of Frances Williams.
  Fact 0.3: The publication is now published by Twin Media Group.
  Fact 0.4: Twin Media Group is owned by Silke Bader.
  Fact 0.5: Twin Media Group is owned by Linda Riley.
Sentence 1: Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.
  Fact 1.0: Elle is a worldwide lifestyle magazine.
  Fact 1.1: Elle is of French origin.
  Fact 1.2: Elle focuses on fashion.
  Fact 1.3: Elle focuses on beauty.
  Fact 1.4: Elle focuses on health.
  Fact 1.5: Elle focuses on entertainment.
Sentence 2:  "Elle" is also the world's best-selling fashion magazine.
  Fact 2.0: Elle is also the world's best-selling fashion magazine.
Sentence 3:  It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.
  Fact 3.0: Elle was founded by Pierre Lazareff.
  Fact 3.1: Elle was founded by Hélène Gordon, the wife of Pierre Lazareff.
  Fact 3.2: Elle was founded in 1945.

Retrieved items:
<item 1>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 1>

<item 2>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 2>

<item 3>
La Belle Assemblée
La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine
published from 1806 to 1837, founded by John Bell (1745–1831).
</item 3>

<item 4>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 4>

<item 5>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 5>

<item 6>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 6>

<item 7>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 7>

<item 8>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 8>

<item 9>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 9>

<item 10>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 10>
=== END ITEM id=266c467dfddc5a27 ===

=== ITEM id=15b66cb907af53af ===
Gold sentences and facts:
Sentence 0:  The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
  Fact 0.0: The documentary explores the history of Sunset Strip.
  Fact 0.1: Sunset Strip is a mile and a half long stretch of road.
  Fact 0.2: Sunset Strip runs through West Hollywood.
Sentence 1:  The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.
  Fact 1.0: The film features some early pioneers of the hardcore punk music scene.
  Fact 1.1: Bad Brains is an early hardcore punk pioneer featured in the film.
  Fact 1.2: Black Flag is an early hardcore punk pioneer featured in the film.
  Fact 1.3: D.O.A. is an early hardcore punk pioneer featured in the film.
  Fact 1.4: Minor Threat is an early hardcore punk pioneer featured in the film.
  Fact 1.5: Minutemen is an early hardcore punk pioneer featured in the film.
  Fact 1.6: SSD is an early hardcore punk pioneer featured in the film.
  Fact 1.7: The film features other early hardcore punk pioneers besides those named.

Retrieved items:
<item 1>
Sunset Strip curfew riots
The Sunset Strip curfew riots, also known as the "hippie riots", were a series of early counterculture-era clashes that took place between police and young
people on the Sunset Strip in Hollywood, California, beginning in the mid-1966 and continuing on and off through the early 1970s.
</item 1>

<item 2>
Sunset Strip (2012 film)
Sunset Strip is a 2012 documentary directed by Hans Fjellestad, and produced by Tommy Alastra.

 The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
</item 2>

<item 3>
Mayor of the Sunset Strip
Mayor of the Sunset Strip is a 2003 documentary film on the life of Rodney Bingenheimer directed by George Hickenlooper, and produced by Chris Carter.

 In 2011, Craig Hlavaty of the "Houston Press" named "Mayor of the Sunset Strip" at number eight on the paper's list of "The 31 Best Music Documentaries of All
Time".

 The film won the Best Documentary Feature at the 2004 Santa Barbara International Film Festival.
</item 3>

<item 4>
I Against I
I Against I is the third studio album by the American hardcore punk band Bad Brains.

 It was released in November 1986 through SST Records with the catalog number SST 065.

 The best-selling album in the band's catalog, "I Against I" is a critically acclaimed album that mixes American hardcore punk with funk, soul, reggae and heavy
metal.

 It is also included in the book "1001 Albums You Must Hear Before You Die".

 The album featured an MTV video for the title track directed by Paul Rachman who later went on to direct the documentary feature film "American Hardcore".
</item 4>

<item 5>
Sunset Strip
Sunset Strip is the mile-and-a-half (2.4 km) stretch of Sunset Boulevard that passes through West Hollywood, California.

 It extends from West Hollywood's eastern border with Hollywood at Crescent Heights Boulevard to its western border with Beverly Hills at Sierra Drive.

 Sunset Strip is probably the best-known portion of Sunset Boulevard, with boutiques, restaurants, rock clubs, and nightclubs.

 It is also known for its array of huge, colorful billboards.
</item 5>

<item 6>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 6>

<item 7>
Riot on Sunset Strip
Riot on Sunset Strip is a 1967 counterculture-era exploitation movie, released by American International Pictures.

 It was filmed and released within four months of the late-1966 Sunset Strip curfew riots.
</item 7>

<item 8>
Vito (film)
Vito is a 2011 documentary produced and directed by Jeffrey Schwarz of the Los Angeles-based production company Automat Pictures.

 The film documents the life of Vito Russo, gay activist, film scholar, and author of "The Celluloid Closet".
</item 8>

<item 9>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 9>

<item 10>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 10>
=== END ITEM id=15b66cb907af53af ===

=== ITEM id=20bd8c7bc53909c7 ===
Gold sentences and facts:
Sentence 0: The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.
  Fact 0.0: The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.
  Fact 0.1: Alberta is a Canadian province.
  Fact 0.2: Alberta is governed under fundamental rules described in the Constitution of Alberta.
Sentence 1:  As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.
  Fact 1.0: Alberta's constitution is an unwritten constitution.
  Fact 1.1: It is typical of all Canadian provinces for their constitutions to be unwritten.
  Fact 1.2: It is typical of Westminster systems more generally for their constitutions to be unwritten.
Sentence 2: The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.
  Fact 2.0: The Westminster system is a parliamentary system of government.
  Fact 2.1: The Westminster system is modelled after the system of government that developed in the United Kingdom.

Retrieved items:
<item 1>
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 1>

<item 2>
Quebec Legislature
The Quebec Legislature (officially Parliament of Quebec, French: "Parlement du Québec" ) is the legislature of the province of Quebec, Canada, since
Confederation in 1867.

 Today, the legislature is made of two elements: the monarch of Canada, represented by the Lieutenant Governor of Quebec, and the unicameral assembly called the
National Assembly of Quebec.

 The legislature has existed since 1867 when Quebec, then called Canada East, became one of the founding colonies of Canadian Confederation.

 From 1867 until 1968 the legislature was bicameral, containing a lower chamber called the Legislative Assembly of Quebec and an upper chamber called the
Legislative Council of Quebec.
</item 2>

<item 3>
The New Canada
The New Canada is a Canadian political literature book written by Reform Party of Canada founder and leader Preston Manning and published by Macmillan Canada.

 The book explains the personal, religious, and political life of Preston Manning and explains the roots and beliefs of the Reform Party.

 At the time of its publishing in 1992, Reform had become a popular populist conservative party in Western Canada after the mainstream Progressive Conservative
Party of Canada was collapsing in support and in 1991 decided to expand eastward into Ontario and the Maritimes.

 One year later the PC party collapsed in the 1993 federal election, allowing the Reform Party to make political history in Canada, deplacing the PCs as the
dominant conservative party in Canada.

 Reform, later renamed the Canadian Alliance, merged with the PC Party in 2003, to form a united right-wing alternative to the governing Liberal Party of
Canada, named the Conservative Party of Canada which has dropped many of the populist themes that the Reform Party had.
</item 3>

<item 4>
Governance of England
In some cases, such as top-up university tuition fees and foundation hospitals, the votes of Scottish MPs have been crucial in helping pass legislation for
England that the majority of English MPs have opposed.
</item 4>

<item 5>
Westminster system
The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.

 This term comes from the Palace of Westminster, the seat of the British parliament.
</item 5>

<item 6>
British North America Acts
The British North America Acts 1867–1975 are the original names of a series of Acts at the core of the constitution of Canada.

 They were enacted by the Parliament of the United Kingdom and the Parliament of Canada.

 In Canada, some of the Acts were amended or repealed by the Constitution Act, 1982.

 The rest were renamed in Canada as the "Constitution Acts".

 In the United Kingdom, those Acts that were passed by the British Parliament remain under their original names.

 The term "British North America" (BNA) refers to the British colonies in North America.
</item 6>

<item 7>
Governance of England
There has not been a government of England since 1707 when the Kingdom of England ceased to exist as a sovereign state, as it merged with the Kingdom of
Scotland to form the Kingdom of Great Britain.

 Kingdom of Great Britain continued from 1707 until 1801 when it merged with the Kingdom of Ireland to form the United Kingdom of Great Britain and Ireland,
which itself became the United Kingdom of Great Britain and Northern Ireland (UK) in 1922 (in reality; in name in 1927) upon independence for most of the island
of Ireland.

 The UK since then has gone through significant change to its system of government, with devolved parliaments, assemblies and governments in Scotland, Wales and
Northern Ireland.

 England, however, remains under the full jurisdiction, on all matters, of the Parliament of the United Kingdom of Great Britain and Northern Ireland and the UK
government as no devolved administration has been created for England within the new structure.

 This situation has led to the anomaly, known as the West Lothian question, which is that Scottish Members of Parliament (MPs) are able to vote on legislation
that affects only England whereas English MPs can not vote on certain Scottish matters due to devolution.
</item 7>

<item 8>
Constitution of Alberta
The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.

 As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.

 Alberta's constitution, like Britain's (on which it is modeled), includes any and all pieces of legislation, court decisions, proclamations, and conventions
which together inform how the province operates.

 Many statutes are important to understanding the governance of the province, but nowhere are they consolidated into a single document or even a list.

 The office of Attorney-General at one time suggested 23 acts which might be included, but cautioned that this was not a "definitive list".

 However, since Alberta is a part of federation, its powers are clearly delineated in law, via the Constitution of Canada.
</item 8>

<item 9>
Canada under British rule
Canada first came under British rule with the Treaty of Paris (1763) which ceded New France, of which Canada was a part, to the British Empire.

 Gradually, other territories, colonies, and provinces that were part of British North America would be added to Canada.

 The Royal Proclamation of 1763 enlarged the colony of Canada under the name of the Province of Quebec, which with the Constitutional Act 1791 became known as
The Canadas.

 With the Act of Union 1840 Upper and Lower Canada were joined to become the United Province of Canada.

 Later, with Confederation in 1867, the British maritime colonies of New Brunswick and Nova Scotia were joined with the British colony of Canada to form the
Dominion of Canada, which was subsequently divided into four provinces, Ontario, Quebec, New Brunswick, and Nova Scotia.

 A number of other British colonies, such as Newfoundland and British Columbia, and large territories such as Rupert's Land initially remained outside of the
newly formed federation.

 Over time, the remaining colonies and territories within British North America came under the control of Canada until the current geographic extent of the
country was reached when Newfoundland and Labrador joined Canada in 1949.
</item 9>

<item 10>
Politics of Canada
The politics of Canada function within a framework of parliamentary democracy and a federal system of parliamentary government with strong democratic
traditions.

 Canada is a constitutional monarchy, in which the Monarch is head of state.

 The country has a multi-party system in which many of its legislative practices derive from the unwritten conventions of and precedents set by the United
Kingdom's Westminster Parliament.

 However, Canada has evolved variations: party discipline in Canada is stronger than in the United Kingdom and more parliamentary votes are considered motions
of confidence, which tends to diminish the role of non-Cabinet Members of Parliament, (MPs).

 Such members, in the government caucus, and junior or lower-profile members of opposition caucuses, are known as backbenchers. Backbenchers can, however, exert
their influence by sitting in parliamentary committees, like the Public Accounts Committee or the National-Defence Committee.
</item 10>
=== END ITEM id=20bd8c7bc53909c7 ===

=== ITEM id=0d4acc6db00d4b93 ===
Gold sentences and facts:
Sentence 0:  It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
  Fact 0.0: The work is the third installment of the 'Naked' series.
  Fact 0.1: 'Naked Killer' is a work in the 'Naked' series released in 1992.
  Fact 0.2: 'Naked Weapon' is a work in the 'Naked' series released in 2002.
Sentence 1: Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
  Fact 1.0: Naked Weapon is also titled 赤裸特工.
  Fact 1.1: Naked Weapon is a 2002 film.
  Fact 1.2: Naked Weapon is a Hong Kong film.
  Fact 1.3: Naked Weapon is an action-thriller film.
  Fact 1.4: Naked Weapon was directed by action choreographer Tony Ching.
  Fact 1.5: Naked Weapon stars Maggie Q.
  Fact 1.6: Naked Weapon stars Anya Wu.
  Fact 1.7: Naked Weapon stars Daniel Wu.

Retrieved items:
<item 1>
Naked Gun 33⅓: The Final Insult
Naked Gun 33⅓: The Final Insult is a 1994 comedy film, and the third and final installment in "The Naked Gun" film series, which was based on the "Police Squad!

" television series.
</item 1>

<item 2>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 2>

<item 3>
Mel Gibson filmography
Five years later he played Martin Riggs in the buddy cop action comedy "Lethal Weapon" alongside Danny Glovera role he later reprised in its sequels "Lethal
Weapon 2" (1989), "Lethal Weapon 3" (1992), and "Lethal Weapon 4" (1998).
</item 3>

<item 4>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 4>

<item 5>
Anatomy 2
Anatomy 2 (German: "Anatomie 2" ) is a 2003 German thriller film written and directed by Stefan Ruzowitzky.

 It's the sequel to the 2000 film "Anatomy", that starred Franka Potente.

 The story moves to Berlin for this film.
</item 5>

<item 6>
Naked Soldier
Naked Soldier (絕色武器) is a 2012 Hong Kong action film directed by Marco Mak and starring Jennifer Tse.

 It is the third installment of the "Naked" series "Naked Killer" (1992) and "Naked Weapon" (2002).
</item 6>

<item 7>
American Pie Presents: The Naked Mile
American Pie Presents: The Naked Mile (also known as American Pie: The Naked Mile) is a 2006 American sex comedy film released by Universal Pictures.

 It is the second installment in the "American Pie Presents" series and the fifth installment in the "American Pie" franchise.

 The film begins a story arc that concludes with "" (2007).

 John White stars as Erik Stifler, a high school senior who is given a "guilt free pass" by his girlfriend, Tracy Sterling (Jessy Schram), and so visits the
Beta House fraternity led by his cousin, Dwight Stifler (Steve Talley), to run a mile naked.

 Christopher McDonald co-stars as Erik's father, Harry, and Eugene Levy once again plays Jim's Dad, who turns out to be a family friend of both Erik's and
Tracy's.

 Also, it is in this film that his name is revealed to be "Noah Levenstein".
</item 7>

<item 8>
The Naked Gun 2½: The Smell of Fear
The Naked Gun 2½: The Smell of Fear is a 1991 comedy film.

 It is the sequel to the 1988 film "" and the second installment in "The Naked Gun" film series.

 The film stars Leslie Nielsen as the comically bumbling Police Lt. Frank Drebin of "Police Squad!

".

 Priscilla Presley plays the role of Jane, with O. J. Simpson as Nordberg and George Kennedy as police captain Ed Hocken.

 The film also features Robert Goulet (who previously made a "special guest star" appearance on "Police Squad!")

 as the villainous Quentin Hapsburg and Richard Griffiths as renewable fuel advocate Dr. Albert S. Meinheimer (as well as his evil double, Earl Hacker).

 Zsa Zsa Gabor, Mel Tormé and members of the Chicago Bears have cameo roles.
</item 8>

<item 9>
Naked Weapon
Naked Weapon (赤裸特工) is a 2002 Hong Kong action-thriller film directed by action choreographer Tony Ching and starring Maggie Q, Anya Wu and Daniel Wu.
</item 9>

<item 10>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 10>
=== END ITEM id=0d4acc6db00d4b93 ===

=== ITEM id=37766f8824f472f8 ===
Gold sentences and facts:
Sentence 0: Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to become lead singer of the hard rock supergroup "Rock Star Supernova".
  Fact 0.0: Lukas Rossi was born on December 21, 1976.
  Fact 0.1: Lukas Rossi is a Canadian rock musician.
  Fact 0.2: Lukas Rossi was the winner of a CBS Television reality series.
  Fact 0.3: That CBS Television reality series was a televised audition contest to become lead singer of the hard rock supergroup 'Rock Star Supernova'.
Sentence 1: Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and founding members of the rock band Chicago.
  Fact 1.0: Terry Alan Kath was born on January 31, 1946.
  Fact 1.1: Terry Alan Kath died on January 23, 1978.
  Fact 1.2: Terry Alan Kath was an American musician.
  Fact 1.3: Terry Alan Kath was an American songwriter.
  Fact 1.4: Terry Alan Kath was best known as the original guitarist of the rock band Chicago.
  Fact 1.5: Terry Alan Kath was one of the lead singers of the rock band Chicago.
  Fact 1.6: Terry Alan Kath was a founding member of the rock band Chicago.

Retrieved items:
<item 1>
Roger Kitter
Roger Kitter (20 October 1949 – 3 January 2015) was an English actor best known for playing Captain Alberto Bertorelli in series 7 of the British sitcom TV
series "'Allo 'Allo!

".

 He had previously appeared weekly with Lulu throughout the 10-week run of her 1973 BBC1 series "It's Lulu".

 With Kaplan Kaye he also recorded a song "Chalk Dust – the Umpire Strikes Back" using the moniker 'The Brat'.

 Released on the Hansa label, it entered the UK Singles Chart on 10 July 1982; it reached a peak of number 19, and remained in the chart for 8 weeks.

 The song was a Top 10 hit in the Netherlands, Belgium and South Africa, and lampooned John McEnroe complaining about line calls in tennis ("The ball's in,
everyone can see that the ball's in!")

.

 He was also an impressionist, who provided the voice of Tommy Cooper in the Lego 'Kipper' advertisement.
</item 1>

<item 2>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 2>

<item 3>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 3>

<item 4>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 4>

<item 5>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 5>

<item 6>
Luther Campbell
Luther Roderick Campbell (born December 22, 1960), also known as Luke Skyywalker and simply Luke, is an American record label owner, rap performer, promoter,
and actor.

 He is best known as a one-time member and leader of rap group 2 Live Crew, and star of his own short-lived show on VH1, "Luke's Parental Advisory".

 As a result of one of the group's songs, which used a parody of Roy Orbison's "Oh, Pretty Woman", Campbell was party to "Campbell v. Acuff-Rose Music, Inc.",
which was argued in front of the U.S. Supreme Court.
</item 6>

<item 7>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 7>

<item 8>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 8>

<item 9>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 9>

<item 10>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 10>
=== END ITEM id=37766f8824f472f8 ===

=== ITEM id=febe693faf920834 ===
Gold sentences and facts:
Sentence 0: The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy Schaal and Henry Gibson.
  Fact 0.0: The 'Burbs is a 1989 American film.
  Fact 0.1: The 'Burbs is a comedy thriller film.
  Fact 0.2: The 'Burbs was directed by Joe Dante.
  Fact 0.3: The 'Burbs stars Tom Hanks.
  Fact 0.4: The 'Burbs stars Bruce Dern.
  Fact 0.5: The 'Burbs stars Carrie Fisher.
  Fact 0.6: The 'Burbs stars Rick Ducommun.
  Fact 0.7: The 'Burbs stars Corey Feldman.
  Fact 0.8: The 'Burbs stars Wendy Schaal.
  Fact 0.9: The 'Burbs stars Henry Gibson.
Sentence 1: Richard "Rick" Ducommun (July 3, 1952 – June 12, 2015) was a Canadian actor, comedian and writer who appeared in films and television.
  Fact 1.0: Rick Ducommun was born on July 3, 1952.
  Fact 1.1: Rick Ducommun died on June 12, 2015.
  Fact 1.2: Rick Ducommun was a Canadian actor.
  Fact 1.3: Rick Ducommun was a Canadian comedian.
  Fact 1.4: Rick Ducommun was a Canadian writer.
  Fact 1.5: Rick Ducommun appeared in films.
  Fact 1.6: Rick Ducommun appeared in television.

Retrieved items:
<item 1>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 1>

<item 2>
Araby Lockhart
Araby Lockhart (born December 4, 1926) is a Canadian stage and television actress, best known for her performances in the films "Capote" and "Police Academy"
and her stage performances as a member of Hart House Theatre and the Straw Hat Players.

 Lockhart has also served as President of the Actors' Fund of Canada.
</item 2>

<item 3>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 3>

<item 4>
Araby Lockhart (born 1926) is a Canadian stage and television actress known for *Capote*, *Police Academy*, Hart House Theatre and the Straw Hat Players; she
also led the Actors’ Fund of Canada. American voice actress Rebecca Forstadt (born 1953), also known as Reba West, voiced young female characters in animated
series. Her career included melodrama at Knott’s Berry Farm, television wardrobe work, Los Angeles theater, and film roles.
</item 4>

<item 5>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 5>

<item 6>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 6>

<item 7>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
</item 7>

<item 8>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 8>

<item 9>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 9>

<item 10>
*Family Matters* was a sitcom about Chicago’s middle-class African American Winslow family. A *Perfect Strangers* spin-off, it aired on ABC (1989–97) and CBS
(1997–98), spanning nine seasons and 215 episodes. Neighbor Steve Urkel (Jaleel White) became its breakout and eventual main character. Kellie Shanygne Williams
(born March 22, 1976) played middle child Laura Winslow; her middle name is pronounced
</item 10>
=== END ITEM id=febe693faf920834 ===
