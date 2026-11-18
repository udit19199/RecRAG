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

=== ITEM id=0f108695e4e4d855 ===
Gold sentences and facts:
Sentence 0:  This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's "Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
  Fact 0.0: This episode is a milestone in the history of the series.
  Fact 0.1: In this episode, Snowball II is killed off.
  Fact 0.2: Homer's mother died in season 19's Mona Leaves-a.
  Fact 0.3: Excluding the death of Homer's mother, the killing of Snowball II is to date the closest thing to an actual Simpson family member being killed off.
Sentence 1: Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".
  Fact 1.0: Mona Penelope Simpson's maiden name is Olsen.
  Fact 1.1: Mona Penelope Simpson is a recurring fictional character.
  Fact 1.2: Mona Penelope Simpson is a character in the animated television series The Simpsons.

Retrieved items:
<item 1>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 1>

<item 2>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 2>

<item 3>
I, (Annoyed Grunt)-bot
"I, (Annoyed Grunt)-Bot", also known as ""I, D'oh-Bot"", is the ninth episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It originally aired on the Fox network in the United States on January 11, 2004.

 This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's
"Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
</item 3>

<item 4>
Good Night (The Simpsons short)
"Good Night" (also known as "Good Night Simpsons") is the first of forty-eight Simpsons shorts that appeared on the variety show "The Tracey Ullman Show".

 It originally aired on the Fox network in the United States on April 19, 1987, during the third episode of "The Tracey Ullman Show" and marks the first
appearance of the Simpson family — Homer, Marge, Bart, Lisa, and Maggie — on television.

 After three seasons on Tracey Ullman, the shorts would be adapted into the animated show "The Simpsons".

 "Good Night" has since been aired on the show in the episode "The Simpsons 138th Episode Spectacular" (in its entirety), along with several other Ullman
shorts, and is one of the few shorts to ever be released on DVD, being included in the Season 1 DVD set.
</item 4>

<item 5>
My Mother the Carjacker
"My Mother the Carjacker" is the second episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It first aired on the Fox network in the United States on November 9, 2003.

 Homer receives a cryptic message in the newspaper informing him to come to a certain place at midnight, and soon discovers that the person who wrote the
message is his mother, Mona Simpson.

 It was written by Michael Price and directed by Nancy Kruse.

 Glenn Close makes her second of six guest spots as Homer's mother.

 It has a direct link from the season seven episode, "Mother Simpson".

 It was nominated for a Writers Guild of America Award in 2004.

 In its original run, the episode received 12.4 million viewers.
</item 5>

<item 6>
Girly Edition
"Girly Edition" is the twenty-first episode in the ninth season of the American animated television series "The Simpsons".

 It originally aired on the Fox network in the United States on April 19, 1998.

 In the episode, Lisa and Bart Simpson must co-anchor a new news program, though when Bart is seen as a more successful news anchor, Lisa becomes jealous and
seeks revenge.

 Meanwhile, in the subplot, Homer Simpson gets a monkey helper because of his laziness.

 "Girly Edition" was the first episode written by Larry Doyle and was directed by Mark Kirkland.

 Much of the subplot was inspired by the film "Monkey Shines".

 Critics gave the episode positive reviews and it was well received by Lisa's voice actress Yeardley Smith.
</item 6>

<item 7>
Mona Leaves-a
"Mona Leaves-a" is the nineteenth episode of "The Simpsons"<nowiki>'</nowiki> nineteenth season.

 It originally aired on the Fox network in the United States on May 11, 2008.

 The episode features the death of Homer's mother, Mona Simpson.

 Homer is reunited with his mother, Mona, but is not willing to forgive her for all the times she left him as a child.

 When she dies, a guilt-ridden Homer attempts to make it up to her by fulfilling her final wishes.

 It was written by Joel H. Cohen and directed by Mike B. Anderson and Ralph Sosa.

 Glenn Close makes her third appearance as Mona Simpson, and Lance Armstrong has a cameo as himself.
</item 7>

<item 8>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 8>

<item 9>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 9>

<item 10>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 10>
=== END ITEM id=0f108695e4e4d855 ===

=== ITEM id=2992ecffb290c5a9 ===
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
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
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
Green Coca-Cola Bottles
With respect to its artistic merits, Warhol’s painting utilizes repetitive imagery in the shape of hundred and twelve almost identical bottles, evoking a sense
of mechanicalization which overlaps his former usage of mass culture objects.

 The widespread display of advertisement flooded the American public during the sixties, and Warhol had been successful in mapping it onto his canvas, no wonder
it is considered one of Warhol’s masterpieces.
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
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 9>

<item 10>
The Andy Warhol Story
The Andy Warhol Story (1966) is an underground film directed by Andy Warhol with cinematography by Paul Morrissey, and starring Edie Sedgwick and Rene Ricard
(as Andy Warhol).
</item 10>
=== END ITEM id=2992ecffb290c5a9 ===

=== ITEM id=db63edec30a742a2 ===
Gold sentences and facts:
Sentence 0: Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.
  Fact 0.0: Francis Keogh Gleason was born on April 14, 1906.
  Fact 0.1: Francis Keogh Gleason died on December 18, 1982.
  Fact 0.2: Francis Keogh Gleason was a resident set decorator at MGM studios.
  Fact 0.3: Francis Keogh Gleason was a resident set decorator at MGM studios for over 40 years.
Sentence 1:  In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi" in 1958) and was nominated an additional 3 times.
  Fact 1.0: Francis Keogh Gleason won 4 Academy Awards during his time at MGM.
  Fact 1.1: Francis Keogh Gleason won an Academy Award for 'An American in Paris' in 1951.
  Fact 1.2: Francis Keogh Gleason won an Academy Award for 'The Bad and the Beautiful' in 1952.
  Fact 1.3: Francis Keogh Gleason won an Academy Award for 'Somebody Up There Likes Me' in 1956.
  Fact 1.4: Francis Keogh Gleason won an Academy Award for 'Gigi' in 1958.
  Fact 1.5: Francis Keogh Gleason was nominated for an Academy Award an additional 3 times.
Sentence 2: Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.
  Fact 2.0: Somebody Up There Likes Me is a 1956 American drama film.
  Fact 2.1: Somebody Up There Likes Me is based on the life of middleweight boxing legend Rocky Graziano.

Retrieved items:
<item 1>
Rocky Graziano
Thomas Rocco Barbella (January 1, 1919 – May 22, 1990), better known as Rocky Graziano, was an American professional boxer who held the World Middleweight
title.

 Graziano is considered one of the greatest knockout artists in boxing history, often displaying the capacity to take his opponent out with a single punch.

 He was ranked 23rd on "The Ring" magazine list of the greatest punchers of all time.

 He fought many of the best middleweights of the era including Sugar Ray Robinson.

 His turbulent and violent life story was the basis of the 1956 Oscar-winning drama film, "Somebody Up There Likes Me", based on his 1955 autobiography of the
same title.
</item 1>

<item 2>
Rocky V
Rocky V is a 1990 American sports drama film.

 It is the fifth film in the "Rocky" series, written by and starring Sylvester Stallone, and co-starring Talia Shire, Stallone's real-life son Sage, and
real-life boxer Tommy Morrison, with Morrison in the role of Tommy Gunn, a talented yet raw boxer.

 Sage played Robert Balboa, whose relationship with his famous father is explored.

 After Stallone directed the second through fourth films in the series, "Rocky V" saw the return of John G. Avildsen, whose direction of "Rocky" won him an
Academy Award for Best Directing.
</item 2>

<item 3>
The Man with the Golden Arm
The Man with the Golden Arm is a 1955 American drama film with elements of film noir, based on the novel of the same name by Nelson Algren, which tells the
story of a drug addict who gets clean while in prison, but struggles to stay that way in the outside world.

 Although the addictive drug is never identified in the film, according to the American Film Institute "most contemporary and modern sources assume that it is
heroin", in contrast to Algren's book which named the drug as morphine.

 The film stars Frank Sinatra, Eleanor Parker, Kim Novak, Arnold Stang and Darren McGavin.

 It was adapted for the screen by Walter Newman, Lewis Meltzer and Ben Hecht (uncredited), and directed by Otto Preminger.

 The film's initial release sparked controversy due to its serious, in-depth treatment of the then-taboo subject of drug addiction.
</item 3>

<item 4>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 4>

<item 5>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 5>

<item 6>
Somebody Up There Likes Me (1956 film)
Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.

 Joseph Ruttenberg was awarded a 1956 Oscar in the category of Best Cinematography (Black and White).

 The film also won the Oscar for Best Art Direction (Cedric Gibbons, Malcolm F. Brown, Edwin B. Willis, F. Keogh Gleason).

 It was directed by Robert Wise.
</item 6>

<item 7>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 7>

<item 8>
Geoffrey Rush
Geoffrey Roy Rush {'1': ", '2': ", '3': ", '4': "} (born 6 July 1951) is an Australian actor and film producer.

 Rush is the youngest amongst the few people who have won the "Triple Crown of Acting": the Academy Award, the Primetime Emmy Award, and the Tony Award.

 He has won one Academy Award for acting (from four nominations), three British Academy Film Awards (from five nominations), two Golden Globe Awards and four
Screen Actors Guild Awards. Rush is the founding President of the Australian Academy of Cinema and Television Arts and was named the 2012 Australian of the
Year.

 He is also the first actor to win the Academy Award, BAFTA Award, Critics' Choice Movie Award, Golden Globe Award, and Screen Actors Guild Award for a single
performance in film for his performance in "Shine" (1996).
</item 8>

<item 9>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 9>

<item 10>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 10>
=== END ITEM id=db63edec30a742a2 ===

=== ITEM id=6f590686cb328a4a ===
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
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 2>

<item 3>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 3>

<item 4>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 4>

<item 5>
Eberhard Köllner
Eberhard Köllner (born 29 September 1939 in Stassfurt, Germany) was selected for Soyuz 31 as the backup for Sigmund Jähn.
</item 5>

<item 6>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 6>

<item 7>
Samantha Cristoforetti
Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.

 She holds the record for the longest uninterrupted spaceflight of a European astronaut (199 days, 16 hours), and until June 2017 held the record for the
longest single space flight by a woman until this was broken by Peggy Whitson.

 She is also the first Italian woman in space.

 Samantha Cristoforetti is also known as the first person who brewed an espresso coffee in space.
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
=== END ITEM id=6f590686cb328a4a ===

=== ITEM id=e663ec512b15d915 ===
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
</item 1>

<item 2>
Todos Los Romances
Todos Los Romances (English: "All the Romances" ) is a three-disc compilation album by Mexican singer Luis Miguel.

 Released on 11 August 1998 by WEA Latina, the record features the three previously released "Romance"-themed albums in which Miguel covered classic boleros in
each of them: "Romance" (1991), "Segundo Romance" (1994), and "Romances" (1997).

 An editor for AllMusic rated the album four of five stars.

 Commercially, "Todos Los Romances" peaked at number four in Spain and was certified double Platinum in the country.

 It also achieved Gold status in Argentina and peaked at number 12 on the "Billboard"'s Top Latin Albums in the United States.
</item 2>

<item 3>
Graph query row:
song: Todos Me Miran
relationship: CREATED_BY
person: Gloria Trevi
occupation: singer
nationality: Mexican
</item 3>

<item 4>
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
</item 4>

<item 5>
Gloria Trevi
Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
</item 5>

<item 6>
No eres tú, soy yo
No eres tú, soy yo (English: It's not you, it's me ) is a 2010 Mexican romantic comedy directed by Alejandro Springall starring Eugenio Derbez, Alejandra Barros
and Martina García and based on the Argentinean film "No sos vos, soy yo" (2004), written by Juan Taratuto.

 It was produced by Matthias Eherenberg and filmed in Mexico City.
</item 6>

<item 7>
Todos Me Miran
Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that
confirmed Trevi's status as a gay icon.

 The song, as interpreted in the music video, is about a young man who dares to crossdress in spite of society's opinions.
</item 7>

<item 8>
Premio Lo Nuestro 2003
In the Regional Mexican, Pilar Montenegro took three awards for Regional Mexican Song of the Year, Pop Song of the Year ("Quitame Ese Hombre"), and for Best
Regional Mexican Female Artist.

 In the tropical genre, Celia Cruz took home with four great awards of the night for Best Salsa Performance, Best Tropical Female Artist, Best Tropical Song of
the year and Tropical Album of the Year.

 At the night, the greatest performance was a medley of top Latin hits from the last 15 years, performed by the artists that made them famous, including Vikki
Carr, Son by Four, Los Ilegales, La Mafia, Luis Enrique, Wilfrido Vargas and Olga Tañón.

 There was a great tribute to Celia Cruz by the world-famous salsa group "Fania All-Stars", of which Cruz was a member during the 1970s, reunited for an
exclusive performance that rocked the house.
</item 8>

<item 9>
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
</item 9>

<item 10>
La Trayectoria (Gloria Trevi album)
La Trayectoria (In English: The Trajectory) is a 2006 double live CD and multimedia DVD release by the Mexican singer Gloria Trevi.

 The DVD includes scenes from parts of the tour from the Palacio de los Deportes of the Mexico City and The Arena Monterrey and the CD includes four new songs
like: "Todos Me Miran" (They All Look At Me) is the first single compilation study of Mexican singer "Gloria Trevi", this is an unreleased song from the album.
</item 10>
=== END ITEM id=e663ec512b15d915 ===

=== ITEM id=78549c2f54c28591 ===
Gold sentences and facts:
Sentence 0: US Airways Flight 1549 was an Airbus A320-214 which, three minutes after takeoff from New York City's LaGuardia Airport on January 15, 2009, struck a flock of Canada geese just northeast of the George Washington Bridge and consequently lost all engine power.
  Fact 0.0: US Airways Flight 1549 was an Airbus A320-214.
  Fact 0.1: US Airways Flight 1549 took off from New York City's LaGuardia Airport.
  Fact 0.2: US Airways Flight 1549 took off on January 15, 2009.
  Fact 0.3: US Airways Flight 1549 struck a flock of Canada geese three minutes after takeoff.
  Fact 0.4: The bird strike happened just northeast of the George Washington Bridge.
  Fact 0.5: US Airways Flight 1549 consequently lost all engine power.
Sentence 1:  Unable to reach any airport, pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River off Midtown Manhattan.
  Fact 1.0: Pilots could not reach any airport.
  Fact 1.1: Pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River.
  Fact 1.2: The Hudson River ditching was off Midtown Manhattan.
Sentence 2: Chesley Burnett "Sully" Sullenberger III (born 1951) is an American retired airline captain celebrated for the January 15, 2009 water landing of US Airways Flight 1549 in the Hudson River off Manhattan after the plane was disabled by striking a flock of Canada geese immediately after takeoff; all 155 people aboard survived.
  Fact 2.0: Chesley Burnett "Sully" Sullenberger III was born in 1951.
  Fact 2.1: Chesley Sullenberger is an American retired airline captain.
  Fact 2.2: Chesley Sullenberger is celebrated for the January 15, 2009 water landing of US Airways Flight 1549 in the Hudson River off Manhattan.
  Fact 2.3: The plane was disabled by striking a flock of Canada geese immediately after takeoff.
  Fact 2.4: All 155 people aboard survived.

Retrieved items:
<item 1>
Chesley Sullenberger
Chesley Burnett "Sully" Sullenberger III (born 1951) is an American retired airline captain celebrated for the January 15, 2009 water landing of US Airways
Flight 1549 in the Hudson River off Manhattan after the plane was disabled by striking a flock of Canada geese immediately after takeoff; all 155 people aboard
survived.

 Sullenberger is an international speaker on airline safety and has helped develop new protocols for airline safety.

 He served as the co-chairman, along with First Officer Jeffrey Skiles, of the EAA's Young Eagles youth introduction-to-aviation program from 2009 to 2013.

 He retired from US Airways after 30 years as a commercial pilot on March 3, 2010.

 In May of the following year, Sullenberger was hired by CBS News as an Aviation and Safety Expert.
</item 1>

<item 2>
US Airline Pilots Association
The US Airline Pilots Association (USAPA) was the collective bargaining agent for the majority former US Airways "East" pilots, and the minority former America
West Airlines "West" pilots.

 USAPA bullied its West pilots, and violated its duty to fairly represent them.

 USAPA was led by East pilot Mike Cleary.

 USAPA was formed by East pilots for the exclusive and immoral - if not illegal - purpose of attempting to renege upon the results of a binding arbitration into
which the East pilots had previously entered with the West pilots.

 After USAPA's dissolution, a number of East activists withheld millions of dollars from the entire membership.

 During the fair representation trial, USAPA used as courtroom witnesses East pilots Sully Sullenberger and Jeffrey Skiles of the infamous Hudson River flight
1549.

 Both witnesses testified in support of the discriminatory practices of USAPA, and why they felt justified in bullying, and attempting to break their arbitrated
agreement with the innocent West pilots.
</item 2>

<item 3>
ALM Flight 980
ALM Antillean Airlines Flight 980 was a flight scheduled to fly from John F. Kennedy International Airport in New York City to Princess Juliana International
Airport in St. Maarten, Netherlands Antilles, on 2 May 1970.

 After several unsuccessful landing attempts, the aircraft's fuel was exhausted and it made a forced water landing (ditching) in the Caribbean Sea 48 km (30 mi)
off St. Croix, with 23 fatalities and 40 survivors.

 The accident is one of a small number of intentional water ditchings of jet airliners.
</item 3>

<item 4>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 4>

<item 5>
Water landing
A water landing is, in the broadest sense, a landing on a body of water.

 Some aircraft such as floatplanes land on water as a matter of course.
</item 5>

<item 6>
1965 Carmel mid-air collision
The 1965 Carmel mid-air collision occurred on December 4, 1965, when Eastern Air Lines Flight 853 (N6218C), a Lockheed Super Constellation en route from Boston
Logan International Airport to Newark International Airport, collided in mid-air with Trans World Airlines Flight 42 (N748TW), a Boeing 707-131B en route from
San Francisco International Airport to John F. Kennedy International Airport, over Carmel, New York, United States.
</item 6>

<item 7>
Wayne W. Lambert
Wayne W. Lambert (born 1936) was an Air force Brigadier General (United States).

 He attended the United States Military Academy at West Point, New York, after already having served a year in the US Army.

 Lambert graduated from the Academy in 1959 and was commissioned a 2nd Lieutenant in the USAF that same year.

 He trained as a bomber pilot, receiving his Aviator badge in 1960.

 He has logged over 5,800 flying hours with the US Air Force.

 A seasoned combat veteran, Lambert flew over 225 combat missions in South East Asia as a B-52 crew member.

 He was awarded the Distinguished Flying Cross (United States) and received 10 awards of the Air Medal for his service during the Vietnamese War.

 After serving in a succession of lesser command positions, Lambert was promoted to the rank of Brigadier General in 1983, assuming command of the former
Strategic Air Command’s 7th Air Division, headquartered then at Ramstein Air Base in West Germany, and overseeing SAC operations throughout the European
theater.

 The Brigadier General retired from the US Air Force in 1989.
</item 7>

<item 8>
Bird strike
A bird strike—sometimes called birdstrike, bird ingestion (for an engine), bird hit, or bird aircraft strike hazard (BASH)—is a collision between an airborne
animal (usually a bird or bat) and a manmade vehicle, especially an aircraft.

 The term is also used for bird deaths resulting from collisions with structures such as power lines, towers and wind turbines (see Bird-skyscraper collisions
and Towerkill).
</item 8>

<item 9>
US Airways Flight 1549
US Airways Flight 1549 was an Airbus A320-214 which, three minutes after takeoff from New York City's LaGuardia Airport on January 15, 2009, struck a flock of
Canada geese just northeast of the George Washington Bridge and consequently lost all engine power.

 Unable to reach any airport, pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River off Midtown Manhattan.

 All 155 people aboard were rescued by nearby boats and there were few serious injuries.
</item 9>

<item 10>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 10>
=== END ITEM id=78549c2f54c28591 ===

=== ITEM id=fa54093ccd2bc89c ===
Gold sentences and facts:
Sentence 0: Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
  Fact 0.0: Martin Archer Flavin was born on November 2, 1883.
  Fact 0.1: Martin Archer Flavin died on December 27, 1967.
  Fact 0.2: Martin Archer Flavin was an American playwright.
  Fact 0.3: Martin Archer Flavin was an American novelist.
Sentence 1: Peter Bradford Benchley (May 8, 1940 – February 11, 2006) was an American author and screenwriter.
  Fact 1.0: Peter Bradford Benchley was born on May 8, 1940.
  Fact 1.1: Peter Bradford Benchley died on February 11, 2006.
  Fact 1.2: Peter Bradford Benchley was an American author.
  Fact 1.3: Peter Bradford Benchley was an American screenwriter.

Retrieved items:
<item 1>
Aaron Barrett
Aaron Asher Barrett (born August 30, 1974) is the lead singer, lead guitarist, and primary songwriter of the American ska-punk band Reel Big Fish.

 Prior to forming Reel Big Fish, Barrett played trombone in another ska-punk act, The Scholars, along with future Reel Big Fish bandmates Scott Klopfenstein and
Grant Barry.
</item 1>

<item 2>
Martin Flavin
Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
</item 2>

<item 3>
Martin Flavin (politician)
Martin Flavin (1841– 30 December 1916) was an Irish nationalist politician, butter merchant and prominent businessman from Cork.

 He was a Member of Parliament (MP) from 1891 to 1892 .
</item 3>

<item 4>
Michael Breed
Michael S. Breed (born May 14, 1962, in Greenwich, Connecticut) is a professional golf instructor and television host.

 In 2003, he was selected as a Top 100 Instructor in America by "Golf Magazine", in 2011, he was voted one of the Top 50 Instructors in America by Golf Digest
(now 13th on this list, and #1 in the state of New York.)

, in 2012, Breed was chosen as the PGA's National Teacher of the Year.

 Prior to his time as a television host, Breed was the Head Golf Professional at Sunningdale Country Club for 12 years from 2001–2012, and the Head Golf
Professional at Birchwood Country Club, as well as the Assistant Professional at Deepdale Golf Club and Augusta National Golf Club.

 He has served on different boards for the Metropolitan PGA for over 10 years and has represented the PGA of America at the National Golf Day in Washington D.C.
since 2012.

 He is involved in charities such as The First Tee, Folds of Honor, Hope for the Warriors, Wounded Warriors Foundation and Salute Military Golf Association.
</item 4>

<item 5>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 5>

<item 6>
Mark Benton
Mark Benton (born 16 November 1965) is an English actor, known for his roles as Eddie in "Early Doors", Howard in "Northern Lights" and Martin Pond in
"Barbara".

 Benton has also starred in BBC One drama, "Waterloo Road" as maths teacher, Daniel Chalk from 2011 to 2013 and again in 2014.
</item 6>

<item 7>
Peter Benchley
Peter Bradford Benchley (May 8, 1940 – February 11, 2006) was an American author and screenwriter.

 He is known for the bestselling novel "Jaws" and co-wrote its subsequent film adaptation with Carl Gottlieb.

 Several more of his works were also adapted for cinema, including "The Deep", "The Island", "Beast", and "White Shark".
</item 7>

<item 8>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 8>

<item 9>
Journey in the Dark
Journey in the Dark is a 1943 novel by Martin Flavin.

 It won both the 1943 Harper Prize and the 1944 Pulitzer Prize.
</item 9>

<item 10>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 10>
=== END ITEM id=fa54093ccd2bc89c ===

=== ITEM id=7b2cfe6ffbd08cf3 ===
Gold sentences and facts:
Sentence 0: Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11, 1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical physics.
  Fact 0.0: Nikolay Mitrofanovich Krylov was born on 29 November [O.S. 17 November] 1879.
  Fact 0.1: Nikolay Mitrofanovich Krylov died on May 11, 1955.
  Fact 0.2: Nikolay Mitrofanovich Krylov's Russian name is Николай Митрофанович Крылов.
  Fact 0.3: Nikolay Mitrofanovich Krylov's Ukrainian name is Микола Митрофанович Крилов.
  Fact 0.4: Nikolay Mitrofanovich Krylov was a Russian mathematician.
  Fact 0.5: Nikolay Mitrofanovich Krylov was a Soviet mathematician.
  Fact 0.6: Nikolay Mitrofanovich Krylov is known for works on interpolation.
  Fact 0.7: Nikolay Mitrofanovich Krylov is known for works on non-linear mechanics.
  Fact 0.8: Nikolay Mitrofanovich Krylov is known for works on numerical methods for solving equations of mathematical physics.
Sentence 1: Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.
  Fact 1.0: Anatoly Timofeevich Fomenko's Russian name is Анатолий Тимофеевич Фоменко.
  Fact 1.1: Anatoly Timofeevich Fomenko was born on 13 March 1945 in Stalino, USSR.
  Fact 1.2: Anatoly Timofeevich Fomenko is a Soviet mathematician.
  Fact 1.3: Anatoly Timofeevich Fomenko is a Russian mathematician.
  Fact 1.4: Anatoly Timofeevich Fomenko is a professor at Moscow State University.
  Fact 1.5: Anatoly Timofeevich Fomenko is well known as a topologist.
  Fact 1.6: Anatoly Timofeevich Fomenko is a member of the Russian Academy of Sciences.

Retrieved items:
<item 1>
Lev Kuleshov (1899–1970), Soviet filmmaker and theorist, pioneered Soviet montage, the Kuleshov effect, and creative geography. Hermann Brunn (1862–1939),
German mathematician, contributed to convex geometry and knot theory. Mercury’s 165-km Chaikovskij crater honors composer Pyotr Tchaikovsky; a 1970 Soviet
biopic about him earned two Oscar nominations. Søren Kierkegaard (
</item 1>

<item 2>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 2>

<item 3>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 3>

<item 4>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 4>

<item 5>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 5>

<item 6>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 6>

<item 7>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 7>

<item 8>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 8>

<item 9>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 9>

<item 10>
The entries profile Russian and Soviet figures across science, arts, politics, and activism: entomologist and prolific author Pavel Marikovsky; film director
Aleksandr Ivanovsky; animator Vladimir Danilevich; actors and collectors Ekaterina and Vladimir Semenikhin; and actor Aleksei Petrenko. They also describe the
anti-tsarist Petrashevsky Circle and member Nikolay Speshnev, plus Marxist revolutionaries, thinkers, and officials including Bazarov, Bag
</item 10>
=== END ITEM id=7b2cfe6ffbd08cf3 ===

=== ITEM id=a92cd812ff3e8153 ===
Gold sentences and facts:
Sentence 0: Wheelock Whitney may refer to one of three members of the Whitney family:
  Fact 0.0: Wheelock Whitney may refer to one of three members of the Whitney family.
Sentence 1: The Whitney family is an American family notable for their social prominence, wealth, business enterprises and philanthropy, founded by John Whitney (1592–1673) who came from London, England to Watertown, Massachusetts in 1635.
  Fact 1.0: The Whitney family is an American family.
  Fact 1.1: The Whitney family is notable for their social prominence, wealth, business enterprises and philanthropy.
  Fact 1.2: The Whitney family was founded by John Whitney (1592–1673).
  Fact 1.3: John Whitney came from London, England.
  Fact 1.4: John Whitney came to Watertown, Massachusetts.
  Fact 1.5: John Whitney came to Watertown, Massachusetts in 1635.

Retrieved items:
<item 1>
Wheelock Whitney
Wheelock Whitney may refer to one of three members of the Whitney family:
</item 1>

<item 2>
John Hay Whitney
John Hay Whitney (August 17, 1904 – February 8, 1982), colloquially known as Jock Whitney, was U.S. Ambassador to the United Kingdom, publisher of the "New York
Herald Tribune", and president of the Museum of Modern Art.

 He was a member of the Whitney family.
</item 2>

<item 3>
Whitney family
The Whitney family is an American family notable for their social prominence, wealth, business enterprises and philanthropy, founded by John Whitney (1592–1673)
who came from London, England to Watertown, Massachusetts in 1635.

 The historic family mansion in Watertown, known as The Elms, was built in 1710.
</item 3>

<item 4>
Charles Andrew Whitney
Charles Andrew Whitney (November 14, 1864- December 31, 1912) was an American businessman and industrialist in the late 19th century, born in Princeton,
Massachusetts.

 He was part of the prominent American Whitney family.

 In 1859, Charles, his brother Levi L. Whitney, and Orville E. Thompson helped lead the large-scale manufacturing of leather boots and shoes in Chicago and were
attributed with successfully running the first factory of its kind there.

 Whitney himself held patents for the manufacture of leather.
</item 4>

<item 5>
Harry Payne Whitney
Harry Payne Whitney (April 29, 1872 – October 26, 1930) was an American businessman, thoroughbred horse breeder, and member of the prominent Whitney family.
</item 5>

<item 6>
Greentree Stable
Greentree Stable, in Red Bank, New Jersey, was a major American thoroughbred horse racing stable and breeding farm established in 1914 by Payne Whitney of the
Whitney family of New York City.

 Payne Whitney operated a horse farm and stable at Saratoga Springs, New York with his brother Harry Payne Whitney, who also had a large stable of horses.

 Greentree Stable had a training base at Aiken, South Carolina, while Greentree Farm in Lexington, Kentucky was established in 1925 as its breeding arm.
</item 6>

<item 7>
James Scollay Whitney
James Scollay Whitney (May 19, 1811 – October 24, 1878) was an American business executive and politician.

 He was the father of Henry Melville Whitney and William Collins Whitney, founders of the Whitney family business interests.
</item 7>

<item 8>
Gertrude Vanderbilt Whitney
Gertrude Vanderbilt Whitney (January 9, 1875 – April 18, 1942) was an American sculptor, art patron and collector, and founder in 1931 of the Whitney Museum of
American Art in New York City.

 She was a prominent social figure and hostess, who was born into the wealthy Vanderbilt family and married into the Whitney family.
</item 8>

<item 9>
Manhasset Stable
The Manhasset Stable was the "nom de course" for an American Thoroughbred horse racing stable established in the early 1930s by Joan Whitney Payson, founder of
the New York Mets baseball team and a member of the prominent New York City Whitney family who have been major figures in the sport for more than one hundred
years.

 Joan Payson named the stable for Manhasset, New York where she grew up.
</item 9>

<item 10>
Roscoe Channing
Roscoe H. Channing, Jr. (January 7, 1868 – April 1, 1961) was an All-American football player, member of the Rough Riders and mining executive.

 Channing was an All-American halfback for Princeton University.

 He was one of eleven players selected by Caspar Whitney for the first ever College Football All-America Team in 1889.

 When the Spanish–American War commenced in 1898, Channing enlisted in Theodore Roosevelt's Rough Riders.

 Roosevelt took pride in how many Ivy League football players enlisted in the Rough Riders.

 Channing later went into the mining business and managed the mining operations of the Whitney family.

 In the 1920s, he formed a partnership with his friend Cornelius Vanderbilt Whitney.

 The two formed the Hudson Bay Mining and Smelting Company in Flin Flon, Saskatchewan, Canada, and Channing served as the company's President.

 Channing died in 1961.
</item 10>
=== END ITEM id=a92cd812ff3e8153 ===

=== ITEM id=c13c315b3cdfce7c ===
Gold sentences and facts:
Sentence 0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
  Fact 0.0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia.
  Fact 0.1: The 1997 European Athletics Junior Championships were held on July 24-27.
Sentence 1: Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.
  Fact 1.0: Ljubljana is the capital of Slovenia.
  Fact 1.1: Ljubljana is the largest city of Slovenia.
  Fact 1.2: Ljubljana is also known by other, historical names.

Retrieved items:
<item 1>
Micom Film and TV Productions
MICOM Film and TV Productions was established in 1989 by Sreco Mihelcic in a Former Yugoslavian republic Slovenia.

 In the early 90' company joined two sons, Matjaz Mihelcic - Program director, Producer and Vasja Mihelcic - technical director and Executive Producer.

 MICOM was in 1990s the leading Slovenian creative house, producing TV shows for youth, popular entertainment TV shows and discovered many new TV concepts.

 Micom discovered many talents, different creative people and gave them possibility to express themselves.

 The most popular TV shows were: Lahkih nog naokrog (educational youth TV series), Cari zacimb (entertaining cooking TV series), Razpoke v casu (fantasy TV
series).

 They also produced documentary movies.

 One of them got in 1996 internationally release - Fight for the river and it's distributed by International Historic Films.

.

 In the beginning of 2000 company joined Marjana Mihelcic.

 It was a family based company with long tradition in show and entertainment business.
</item 1>

<item 2>
1964 European Junior Games
The 1964 European Junior Games was the first edition of the biennial athletics competition for European athletes aged under twenty.

 It was an unofficial competition without sanction from the European Athletic Association.

 The event was held at the 10th-Anniversary Stadium in Warsaw, Poland.

 The competition eventually led to the creation of the European Athletics Junior Championships in 1970.
</item 2>

<item 3>
Eliška Drahotová
Eliška Drahotová (] ; born 22 July 1995, Rumburk, Czech Republic) is a Czech athletics competitor in racewalking and road cycling, like her twin sister, Anežka.

 She won the bronze medal in the 2013 European Athletics Junior Championships, while her twin sister Anežka Drahotová took the gold.

 Her global debut in cycling also came that year at the 2013 UCI Road World Championships.

 There she placed 34th in the junior time trial.
</item 3>

<item 4>
Ljubljana
Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.

 It has been the cultural, educational, economic, political, and administrative center of independent Slovenia since 1991.

 Its central geographic location within Slovenia, transport connections, concentration of industry, scientific and research institutions, and cultural tradition
are contributing factors to its leading position.
</item 4>

<item 5>
2015 European Athletics Junior Championships
The 2015 European Athletics Junior Championships were the 23rd edition of the biennial European U20 athletics championships.

 They were held in Eskilstuna, Sweden from 16 July to 19 July.
</item 5>

<item 6>
1997 European Athletics Junior Championships
The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
</item 6>

<item 7>
Amela Terzić
Amela Terzić (Serbian Cyrillic: Амела Терзић, born 2 January 1993 in Priboj, FR Yugoslavia) is a Serbian middle-distance runner.

 She won two gold medals at 2011 European Athletics Junior Championships in Tallinn and was the junior champion at the 2012 European Cross Country
Championships.

 She has also been a medallist in the 1500 metres at the World Junior and World Youth Championships and a gold medalist at the 2013 European U23 Championship.
</item 7>

<item 8>
European Athletics U20 Championships
The European Athletics U20 Championships (formerly named the European Athletics Junior Championships up to 2015) are the European championships for athletes who
are 19 years of age or under, which is the age range recognised by the IAAF as junior athletes.

 The event is currently organized by the European Athletic Association.
</item 8>

<item 9>
Ioan Silviu Suciu
Ioan Silviu Suciu (born 24 November 1977 in Sibiu) is a retired Romanian artistic gymnast.

 His best event was the pommel horse.

 One of his closest rivals was his team mate and pommel horse Olympic champion Marius Urzică, who was defeated by Suciu in Ljubljana, 2004, when the latter won
the title while Urzică took 8th place.

 Suciu is an Olympic bronze medalist with the team, a silver world medalist on pommel horse and a six-time European medalist (pommel horse, vault and team).

 Suciu was one of the key team members of the Romanian gymnastics team for several years contributing to the 2004 Olympic team bronze medal and three
continental team medals (two gold and one silver).

 He also placed fourth all around at the 2004 Olympic Games.
</item 9>

<item 10>
1968 European Junior Games
The 1968 European Junior Games was the third edition of the biennial athletics competition for European athletes aged under twenty.

 It was the final edition of the competition under that name and was succeeded by the European Athletics Junior Championships in 1970.

 The event was held in Leipzig, part of the German Democratic Republic at that time.

 The competition was mainly between Eastern European countries, with Belgium, Austria, Netherlands and Greece being present from the west.
</item 10>
=== END ITEM id=c13c315b3cdfce7c ===

=== ITEM id=aa43ba178b2ddb97 ===
Gold sentences and facts:
Sentence 0:  Պողոսյան , born July 15, 1990) is an American professional ice hockey defenseman currently playing for the Buffalo Sabres.
  Fact 0.0: Պողոսյան was born on July 15, 1990.
  Fact 0.1: Պողոսյան is an American professional ice hockey defenseman.
  Fact 0.2: Պողոսյան is currently playing for the Buffalo Sabres.
Sentence 1: Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record producer.
  Fact 1.0: The Armenian name of Daron Vartan Malakian is Տարօն/Տարոն Վարդան Մալաքեան.
  Fact 1.1: Daron Vartan Malakian was born on July 18, 1975.
  Fact 1.2: Daron Vartan Malakian is an Armenian-American singer-songwriter.
  Fact 1.3: Daron Vartan Malakian is a multi-instrumentalist.
  Fact 1.4: Daron Vartan Malakian is a record producer.

Retrieved items:
<item 1>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 1>

<item 2>
Graph query row:
person: Daron Malakian
occupation: ['singer-songwriter', 'multi-instrumentalist', 'record producer', 'guitarist', 'songwriter', 'vocalist']
</item 2>

<item 3>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 3>

<item 4>
Dave Kneebone
Dave Kneebone is an American producer.

 Along with Eric Wareheim and Tim Heidecker he runs Abso Lutely Productions which produces a variety of television and films.

 His role has been described as the business chief and "straight man" at Abso Lutely.

 Kneebone has worked as a producer on a variety of television shows including "Comedy Bang!

 Bang!

", "Nathan for You", and "Tim and Eric Awesome Show, Great Job!

".

 He is also listed a producer on the feature film "Tim and Eric's Billion Dollar Movie".
</item 4>

<item 5>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 5>

<item 6>
Zach Bogosian
In his second season, he tied a team record for goals by a defenseman.

 Bogosian first played in an international tournament when he joined the American national team at the 2009 IIHF World Championship.
</item 6>

<item 7>
Bad Acid Trip
Bad Acid Trip is an American metal band, currently signed to System of a Down lead singer Serj Tankian's record label, Serjical Strike Records.

 Their 2004 album, "Lynch the Weirdo", was produced by Daron Malakian, another System of a Down member.

 In 2006 they performed on the second stage at the Ozzfest summer tour.

 The band has three full-length releases ("For the Weird by the Weird", "Lynch the Weirdo", and "Humanly Possible") and multiple EPs.
</item 7>

<item 8>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 8>

<item 9>
Zach Bogosian
Zachary M. Bogosian (Armenian: Զաքարի Մ.

 Պողոսյան , born July 15, 1990) is an American professional ice hockey defenseman currently playing for the Buffalo Sabres.

 Bogosian attended Cushing Academy in Massachusetts before he joined the Peterborough Petes of the Ontario Hockey League (OHL) in 2006.

 He played two seasons in the OHL, and was nominated for the Red Tilson Trophy as the league's most outstanding player in his second season.

 Bogosian was regarded as a complete, physical defenseman who could contribute on both offense and defense; he was rated as one of the top players heading into
the 2008 NHL Entry Draft, where the Atlanta Thrashers selected him third overall.

 He signed a contract with the team a few weeks after the draft and began the season with the Thrashers, though he missed several weeks of his rookie year due
to injury.

 In his second season, he tied a team record for goals by a defenseman.
</item 9>

<item 10>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 10>
=== END ITEM id=aa43ba178b2ddb97 ===
