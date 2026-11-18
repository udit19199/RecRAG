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

=== ITEM id=e043c5ce69c451b7 ===
Gold sentences and facts:
Sentence 0:  It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of Superman called Nuclear Man.
  Fact 0.0: The film is the fourth and final film in the original Superman film series.
  Fact 0.1: The film is the first film in the original Superman series not produced by Alexander and Ilya Salkind.
  Fact 0.2: The film was produced by Golan-Globus' Cannon Films.
  Fact 0.3: The film was produced in association with Warner Bros.
  Fact 0.4: Gene Hackman returned as Lex Luthor in the film.
  Fact 0.5: Lex Luthor creates an evil solar-powered version of Superman called Nuclear Man.
Sentence 1: Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.
  Fact 1.0: Eugene Allen 'Gene' Hackman was born on January 30, 1930.
  Fact 1.1: Gene Hackman is a retired American actor.
  Fact 1.2: Gene Hackman is a retired American novelist.

Retrieved items:
<item 1>
Alternative versions of Lex Luthor
The fictional character Lex Luthor has appeared in a number of media, always as the archenemy of Superman.

 Each version of the work typically establishes its own continuity, and sometimes introduces parallel universes, to the point where distinct differences in the
portrayal of the character can be identified.

 In addition, the DC Universe has been rewritten a number of times, establishing additional versions of the character.

 This article details and lists various versions of Lex Luthor depicted in works including DC Comics "Multiverse", "Elseworlds", television and film.
</item 1>

<item 2>
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 2>

<item 3>
Lex Luthor (Smallville)
Lex Luthor is a fictional character from the television series "Smallville".

 He was a series regular from the pilot episode until the season seven finale, and has been played continuously by Michael Rosenbaum, with various actors
portraying the character as a child throughout the series.

 The character of Lex Luthor, first created for comic books by Jerry Siegel and Joe Shuster in 1940 as nemesis of Superman, was adapted to television in 2001 by
Alfred Gough and Miles Millar—this is only the third time the character has been adapted to a live action television series.

 The character has also appeared in various literature based on the "Smallville" television series, none of which directly continues from or into the television
episodes.
</item 3>

<item 4>
The entries cover DC stories and adaptations centered on Batman, Superman, and Lex Luthor. The 2017 animated *Batman vs. Two-Face* reunites Adam West, Burt
Ward, and Julie Newmar in their 1960s TV roles. Dennis O’Neil and Dick Giordano’s *The Man Who Falls* traces Bruce Wayne’s origins and influenced *Batman
Begins*. Superman works include *Superman IV*, *Superman Returns*, Tom De Haven’s 1930
</item 4>

<item 5>
Batman v Superman: Dawn of Justice
Batman v Superman: Dawn of Justice is a 2016 American superhero film featuring the DC Comics characters Batman and Superman.

 Directed by Zack Snyder and distributed by Warner Bros.

 Pictures, the film is the second installment in the DC Extended Universe (DCEU) following 2013's "Man of Steel".

 It was written by Chris Terrio and David S. Goyer, and features an ensemble cast that includes Ben Affleck, Henry Cavill, Amy Adams, Jesse Eisenberg, Diane
Lane, Laurence Fishburne, Jeremy Irons, Holly Hunter and Gal Gadot.

 "Batman v Superman: Dawn of Justice" is the first live-action film to feature Batman and Superman together, as well as the first live-action cinematic
portrayal of Wonder Woman, Aquaman, Flash and Cyborg.

 In the film, criminal mastermind Lex Luthor manipulates Batman into a preemptive battle with Superman, whom Luthor is obsessed with defeating.
</item 5>

<item 6>
Superman IV: The Quest for Peace
Superman IV: The Quest for Peace is a 1987 American superhero film directed by Sidney J. Furie, based on the DC Comics character Superman.

 It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but
rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of
Superman called Nuclear Man.
</item 6>

<item 7>
Superman: A Tale of Five Cities
Superman og Fredsbomben (translation: Superman and the Peace Bomb) or Superman: A Tale of Five Cities is a large prestige format 48-page graphic novel published
by Danish publisher Interpresse in 1990.

 In celebration of Superman's 50th anniversary in 1988, DC Comics incentived overseas publishers to produce an original story with the Man of Steel.

 Only the Interpresse editors from Denmark attended the call, comic book translator and writer Niels Søndergaard wrote the story featuring Clark Kent and Lois
Lane in a cold war adventure where Superman needs to devise a Lex Luthor plan to control all Nuclear Bombs in Europe.

 The storyline presents the Man of Steel visiting five capitals from northern Europe and Scandinavia: Amsterdam, Copenhagen, Oslo, Stockholm and Helsinki.

 The art and colors are provided by Teddy Kristiansen and the lettering by Rebecca Løwe.

 It was dedicated to danish editors Henning Kure and Ove Høyer, pioneers of the super-heroes comics in Denmark.
</item 7>

<item 8>
It's Superman!
It's Superman!

 is a novel by Tom De Haven based on the comic book superhero Superman.

 It was released on September 15, 2005 in hardcover and August 29, 2006 in paperback.

 The premise tells the tale of Clark Kent's beginnings into becoming Superman, set in the 1930s, where Clark befriends a wrongly convicted photographer named
Willi Berg, and is then taken from Kansas to Hollywood and finally in New York where he meets Lois Lane, fights Lex Luthor, as he debuts in his superhero
persona.

 Despite the setting, this is not about the Golden Age Superman also known as the Superman of Earth-2; as Perry White, the Daily Planet, Lex Luthor's position
and his trademark powers (including flight) are not part of that alternate Earth.

 Rather, it's a Superman period piece set in the 1930s.
</item 8>

<item 9>
Lex Luthor: Man of Steel
Lex Luthor: Man of Steel (later collected as simply Luthor) is a five-issue monthly American comic book limited series written by Brian Azzarello and
illustrated by Lee Bermejo, which features Superman's nemesis Lex Luthor as the protagonist.
</item 9>

<item 10>
Lionel Luthor
Lionel Luthor is a fictional character portrayed by John Glover in the television series "Smallville".

 The character was initially a recurring guest in season one, and became a series regular in season two and continued until being written out of the show in
season seven.

 The character returned to the show in season ten again as a recurring guest role as a parallel universe (Earth-2) version of the character.

 In "Smallville", Lionel Luthor is the father of Lex Luthor (Michael Rosenbaum), and founder and CEO of LuthorCorp.

 Lex Luthor's father was first introduced in "Superman" comics by Jerry Siegel in 1961 and has since appeared in other Superman-related media under different
names.

 "Smallville" is the first appearance in which the character has been an intricate part of a Superman adaptation.

 Series developers Alfred Gough and Miles Millar created Lionel Luthor for "Smallville" to provide an antithesis to the parenting style of Jonathan Kent (John
Schneider) and Martha Kent (Annette O'Toole).
</item 10>
=== END ITEM id=e043c5ce69c451b7 ===

=== ITEM id=ba111342a2955b5b ===
Gold sentences and facts:
Sentence 0: David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
  Fact 0.0: David Ross Lederman was born December 12, 1894.
  Fact 0.1: David Ross Lederman died August 24, 1972.
  Fact 0.2: David Ross Lederman was an American film director.
  Fact 0.3: David Ross Lederman was noted for his Western, action, and adventure films of the 1930s and 1940s.
Sentence 1: Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
  Fact 1.0: Stefan Ruzowitzky was born 25 December 1961.
  Fact 1.1: Stefan Ruzowitzky is an Austrian film director.
  Fact 1.2: Stefan Ruzowitzky is an Austrian screenwriter.

Retrieved items:
<item 1>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 1>

<item 2>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 2>

<item 3>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 3>

<item 4>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 4>

<item 5>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 5>

<item 6>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 6>

<item 7>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 7>

<item 8>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 8>

<item 9>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 9>

<item 10>
Wolfgang Becker
Wolfgang Becker (born 22 June 1954) is a German film director and writer.

 He is best known to the international audience for his work "Good Bye Lenin!

" (2003).
</item 10>
=== END ITEM id=ba111342a2955b5b ===

=== ITEM id=c5c41a9998635498 ===
Gold sentences and facts:
Sentence 0: 3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.
  Fact 0.0: 3:10 to Yuma is a 2007 American Western film.
  Fact 0.1: 3:10 to Yuma was directed by James Mangold.
  Fact 0.2: 3:10 to Yuma was produced by Cathy Konrad.
  Fact 0.3: Russell Crowe and Christian Bale star in the lead roles of 3:10 to Yuma.
  Fact 0.4: Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman have supporting performances in 3:10 to Yuma.
Sentence 1: Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and Wat in "A Knight's Tale".
  Fact 1.0: Alan Wray Tudyk was born on March 16, 1971.
  Fact 1.1: Alan Tudyk is an American actor.
  Fact 1.2: Alan Tudyk is an American voice actor.
  Fact 1.3: Alan Tudyk is known for playing Hoban "Wash" Washburne in the space western television series "Firefly" and in the movie "Serenity".
  Fact 1.4: Alan Tudyk is known for playing Alpha in the science fiction TV series "Dollhouse".
  Fact 1.5: Alan Tudyk is known for playing Tucker McGee in "Tucker & Dale vs. Evil".
  Fact 1.6: Alan Tudyk is known for playing Steve the Pirate in "" (the title is missing from the source sentence).
  Fact 1.7: Alan Tudyk is known for playing Wat in "A Knight's Tale".

Retrieved items:
<item 1>
Henry Murdoch
Henry Murdoch (17 September, 1920 - 24 April, 1987), born as George Henry Murdock, was an Australian aboriginal actor and stockman who appeared in Australian
films of the 1940s and 1950s.

 He was working as stockman in Rockhampton when discovered by Ralph Smart, who was helping make "The Overlanders" (1946).

 The film's director, Harry Watt, later claimed Murdoch and fellow aboriginal actor Clyde Combo "proved to be first-class actors and were exceedingly quick
witted and intelligent.

 They certainly disproved the conventional idea that the Australian aboriginal is an animalistic caveman."
</item 1>

<item 2>
3:10 to Yuma (2007 film)
3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead
roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.

 It is about a drought-impoverished rancher (Bale) who takes on the dangerous job of taking a notorious outlaw (Crowe) to justice.

 It is a remake of the , making it the second adaptation of Elmore Leonard's short story "Three-Ten to Yuma".

 Filming took place in various locations in New Mexico.

 "3:10 to Yuma" opened September 7, 2007, in the United States and received positive reviews from critics.
</item 2>

<item 3>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 3>

<item 4>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 4>

<item 5>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 5>

<item 6>
Richard T. Jones
Richard Timothy Jones (born January 16, 1972), known professionally as Richard T. Jones, is an American actor.

 Jones is best known for his portrayals of Laveinio in the dramatic film "The Wood" and Mike of the dramatic films "Why Did I Get Married?

" and "Why Did I Get Married Too?

".
</item 6>

<item 7>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 7>

<item 8>
Deadfall (2012 film)
Deadfall is a 2012 American crime drama film directed by Stefan Ruzowitzky, written by Zach Dean and starring Eric Bana, Olivia Wilde, and Charlie Hunnam.
</item 8>

<item 9>
Alan Tudyk
Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television
series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and
Wat in "A Knight's Tale".

 He also co-starred on the ABC sitcom "Suburgatory".

 Tudyk had a supporting role in the film "42" as baseball manager Ben Chapman and voiced roles in every Walt Disney Animation Studios film to date starting with
"Wreck-It Ralph".

 He has also featured in the web series "Con Man" and the films "" and "Rogue One: A Star Wars Story".
</item 9>

<item 10>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 10>
=== END ITEM id=c5c41a9998635498 ===

=== ITEM id=287e937b430b417f ===
Gold sentences and facts:
Sentence 0:  He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his early work as Knox Overstreet in "Dead Poets Society".
  Fact 0.0: The actor referred to as 'He' is best known for the role of Dan Rydell on "Sports Night."
  Fact 0.1: The actor referred to as 'He' is best known for the role of Will Gardner on "The Good Wife."
  Fact 0.2: The role of Will Gardner on "The Good Wife" earned the actor two Primetime Emmy Award nominations.
  Fact 0.3: The actor referred to as 'He' is best known for his early work as Knox Overstreet in "Dead Poets Society."
Sentence 1: Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.
  Fact 1.0: Dead Poets Society is a 1989 American drama film.
  Fact 1.1: Dead Poets Society was directed by Peter Weir.
  Fact 1.2: Dead Poets Society was written by Tom Schulman.
  Fact 1.3: Dead Poets Society stars Robin Williams.
Sentence 2:  Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students through his teaching of poetry.
  Fact 2.0: Dead Poets Society is set in 1959.
  Fact 2.1: Dead Poets Society is set at the fictional elite conservative Vermont boarding school Welton Academy.
  Fact 2.2: Dead Poets Society tells the story of an English teacher.
  Fact 2.3: The English teacher in Dead Poets Society inspires his students through his teaching of poetry.

Retrieved items:
<item 1>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 1>

<item 2>
Dead Poets Society
Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.

 Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students
through his teaching of poetry.
</item 2>

<item 3>
Meet the Robinsons
Meet the Robinsons is a 2007 American computer-animated science fiction comedy film produced by Walt Disney Animation Studios and released by Walt Disney
Pictures on March 30, 2007.

 The 47th Disney animated feature film, it was released in standard and Disney Digital 3-D versions.

 The film is loosely based on characters from the children's book "A Day with Wilbur Robinson", by William Joyce.

 The voice cast includes Jordan Fry, Wesley Singerman, Harland Williams, Tom Kenny, Steve Anderson, Laurie Metcalf, Adam West, Tom Selleck, and Angela Bassett.

 It was the first film released after John Lasseter became chief creative officer of Walt Disney Animation Studios.
</item 3>

<item 4>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 4>

<item 5>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 5>

<item 6>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 6>

<item 7>
Josh Charles
Joshua Aaron "Josh" Charles (born September 15, 1971) is an American stage, film and television actor.

 He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his
early work as Knox Overstreet in "Dead Poets Society".
</item 7>

<item 8>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 8>

<item 9>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 9>

<item 10>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 10>
=== END ITEM id=287e937b430b417f ===

=== ITEM id=1d04335cf5256ab0 ===
Gold sentences and facts:
Sentence 0: Baghdad ER is a documentary released by HBO on May 21, 2006.
  Fact 0.0: Baghdad ER is a documentary.
  Fact 0.1: Baghdad ER was released by HBO.
  Fact 0.2: Baghdad ER was released on May 21, 2006.
Sentence 1: Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium in Berlin, Germany.
  Fact 1.0: Olympia is a documentary film.
  Fact 1.1: Olympia was released in 1938.
  Fact 1.2: Olympia is a German film.
  Fact 1.3: Olympia was written by Leni Riefenstahl.
  Fact 1.4: Olympia was directed by Leni Riefenstahl.
  Fact 1.5: Olympia was produced by Leni Riefenstahl.
  Fact 1.6: Olympia documents the 1936 Summer Olympics.
  Fact 1.7: The 1936 Summer Olympics were held in the Olympic Stadium in Berlin, Germany.

Retrieved items:
<item 1>
Baghdad ER
Baghdad ER is a documentary released by HBO on May 21, 2006.

 It shows the Iraq war from the perspective of a military hospital in Baghdad.

 It has some relatively disturbing scenes in it (e.g. amputations), therefore the U.S. Army is officially warning that military personnel watching it could
experience symptoms of post-traumatic stress disorder (PTSD).
</item 1>

<item 2>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 2>

<item 3>
Matthew O'Neill (filmmaker)
Matthew O'Neill is a documentary filmmaker best known for his work on the HBO film "Baghdad ER", for which he and co-creator Jon Alpert won three Emmy Awards.
</item 3>

<item 4>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 4>

<item 5>
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
</item 5>

<item 6>
Baghdad Airport Road
The Baghdad Airport Road is a 12-kilometer (7.5 mi) stretch of highway in Baghdad, Iraq linking the Green Zone, a heavily fortified area at the center of
Baghdad, to Baghdad International Airport (BIAP).

 It also links different parts of Baghdad to the Airport and connects neighboring areas to each other.

 It became prominent after the 2003 invasion of Iraq following its capture by the Coalition Forces.

 Although it was commonly referred to by the military Main Supply Route (MSR) designation "Route Irish", the route from the International Zone to the airport
stretches over two MSRs: Route Aeros, the section leading into and out of the International Zone, and Route Irish, which stretches east from the airport, then
turns south (past the junction with Route Aeros) to a junction with Highway 1 (MSR Tampa).
</item 6>

<item 7>
Hysterical Blindness (film)
Hysterical Blindness is a TV movie made for HBO, directed by Mira Nair and starring Gena Rowlands, Uma Thurman, Juliette Lewis and Ben Gazzara.

 The movie premiered on HBO on August 21, 2002 to good reviews.

 In 2003, Uma Thurman won a Golden Globe Award for her portrayal of Debby Miller.

 Ben Gazzara and Gena Rowlands also won Best Supporting Actor/Actress awards for their performances as Virginia Miller and Nick Piccolo at the 2003 Emmy Awards.
</item 7>

<item 8>
Sarajevo (1940 French film)
Sarajevo (French:De Mayerling à Sarajevo) is a 1940 French historical film directed by Max Ophüls and starring Edwige Feuillère, John Lodge and Aimé Clariond.

 Beginning in the aftermath of the Mayerling Incident the film portrays the love affair and marriage between Archduke Franz Ferdinand of Austria and Sophie,
Duchess of Hohenberg, leading up to their eventual assassination in 1914 in events that triggered the First World War.

 The film was not a commercial or critical success.

 Following the German occupation of France the film was banned, and Ophüls fled into exile for the second time.
</item 8>

<item 9>
Baghdad Jewish Arabic
Baghdad Jewish Arabic is the Arabic dialect spoken by the Jews of Baghdad and other towns of Southern Iraq.

 This dialect differs from the dialect spoken by the Jews in Northern Iraq, such as Mosul and 'Ana.

 The Baghdadi and Northern dialects may be regarded as subvarieties of Judeo-Iraqi Arabic.

 As with most Judeo-Arab communities, there are likely to be few, if any, speakers of the Judeo-Iraqi Arabic dialects who still reside within Iraq.

 Rather these dialects have been maintained or are facing critical endangerment within respective Judeo-Iraqi diasporas, namely those of Israel and the United
States.

 In 2014, the film "Farewell Baghdad" (Arabic: مطير الحمام; Hebrew: מפריח היונים, lit.

 "The Dove Flyer"), which is performed mostly in Jewish Baghdadi Arabic dialect, became the first film to be almost completely performed in Judeo-Iraqi Arabic.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=1d04335cf5256ab0 ===

=== ITEM id=fd4751e80dcd2c1f ===
Gold sentences and facts:
Sentence 0:  The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.
  Fact 0.0: The album includes a collaboration with Popcaan.
  Fact 0.1: The album includes a collaboration with Zhu.
  Fact 0.2: The album includes a collaboration with Leikeli47.
  Fact 0.3: The album includes a collaboration with Dreezy.
  Fact 0.4: The album includes a collaboration with Flume.
  Fact 0.5: The album includes a collaboration with Pell.
Sentence 1: Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of 2014, signed to Mind of a Genius Records and Columbia Records.
  Fact 1.0: Steven Zhu was born in 1989.
  Fact 1.1: Steven Zhu is known professionally as Zhu.
  Fact 1.2: Zhu is a Chinese American electronic musician.
  Fact 1.3: Zhu is a Chinese American singer.
  Fact 1.4: Zhu has been active since the beginning of 2014.
  Fact 1.5: Zhu is signed to Mind of a Genius Records.
  Fact 1.6: Zhu is signed to Columbia Records.

Retrieved items:
<item 1>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 1>

<item 2>
Kid Confucius
Kid Confucius were an eight-piece Australian band from Sydney, Australia.

 They formed in 2001 and have since played well over 400 live shows around the country, including major festivals as well as their own headline shows at venues
such as The Annandale Hotel and The Metro Theatre.

 In 2005 the band released a self-titled album and two singles "Words" and "Skintight."

 The album, a mish-mash of soul, hip-hop, pop and funk, received great critical acclaim, most notably from Rolling Stone who hailed the album as one of the
standout local releases of the year.

 "Words" also enjoyed some solid months of radio and TV play.

 The band released its second album Stripes in 2007 with three singles "Closer", "Last Straw" and "Moment".

 Stripes was the band's attempt at making a Detroit-era Motown soul album and it was quick to earn rave reviews from press around the country as well as a
publishing deal with Mushroom.

 "Moment" has enjoyed good radio play on triple j. Kid Confucius is set to release its third album in October 2008.

 The first single from this album, "Good Luck", is out now.
</item 2>

<item 3>
Candy Chen
Candy Chen (陳斯亞 born January 3, 1993) is a dancer, actress, host, singer, rapper and a model.

 She is featured on the Taiwanese variety shows: Blackie's Teenage Club and Blackie Lollipop.

 Candy was a member of the Taiwanese girl group Hey Girl from 2010 to 2011, and a member of from 2013 to 2016.

 She is 1/8th Dutch.

 She changed her name to Chen Yi-Ling in July 2012 to follow her mother's maiden name.

 Her English name becomes Nina Chen.

 In November 2013, she changed her name to Candy Chen (English Name) and Chen Si Ya (Chinese Name).
</item 3>

<item 4>
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
</item 4>

<item 5>
Oliver Lieb
Oliver Lieb (born 1969 in Frankfurt, Germany) is a German electronic music producer and DJ.

 Lieb is known to have more than a dozen aliases with over 200 productions and remixes in various electronic genres such as trance, house, and techno.

 He has also produced tracks for other artists.
</item 5>

<item 6>
King Electric
King Electric is Chris Isepp and Peter Hartwig.

 They took their first musical attempts at the end of 1980s / beginning 1990s and first met each other at the upcoming and flourishing Viennese techno scene
back in '92.

 While Hartwig was touring with the well known noise-pop band the Basket Boys, Isepp made first attentions as Electro DJ & Producer.
</item 6>

<item 7>
I Remember (AlunaGeorge album)
I Remember is the second studio album by English electronic music duo AlunaGeorge.

 It was released on 16 September 2016 by Island Records.

 The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.

 The album's lead single, "I'm in Control", was released in February 2016, featuring vocals from rapper Popcaan, and peaked at number 39 on the UK Singles
Chart.
</item 7>

<item 8>
Raleigh Moncrief
Raleigh Moncrief is an American electronic musician, record producer and recording engineer who has worked with Zach Hill, Dirty Projectors and !!!

.
</item 8>

<item 9>
Zhu (musician)
Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of
2014, signed to Mind of a Genius Records and Columbia Records.

 Until mid-2014, Zhu remained anonymous, asking to be judged by his music alone.

 His debut album, "Generationwhy", was released on 29 July 2016.
</item 9>

<item 10>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 10>
=== END ITEM id=fd4751e80dcd2c1f ===

=== ITEM id=fb1aabfa279d0448 ===
Gold sentences and facts:
Sentence 0:  The Park is home to the first flume ride in Ireland (Pirates Plunge).
  Fact 0.0: The Park is home to the first flume ride in Ireland.
  Fact 0.1: The first flume ride in Ireland is named Pirates Plunge.
Sentence 1: Grace O'Malley (c. 1530 – c. 1603; also Gráinne O'Malley, Irish: "Gráinne Ní Mháille" ) was lord of the Ó Máille dynasty in the west of Ireland, following in the footsteps of her father Eoghan Dubhdara Ó Máille.
  Fact 1.0: Grace O'Malley lived from about 1530 to about 1603.
  Fact 1.1: Grace O'Malley is also called Gráinne O'Malley.
  Fact 1.2: Grace O'Malley's Irish name is Gráinne Ní Mháille.
  Fact 1.3: Grace O'Malley was lord of the Ó Máille dynasty in the west of Ireland.
  Fact 1.4: Grace O'Malley followed in the footsteps of her father, Eoghan Dubhdara Ó Máille.

Retrieved items:
<item 1>
Pirate Adventure Park
Pirate Adventure Park is a small theme park located beside Westport House in Westport, Ireland.

 The Park is home to the first flume ride in Ireland (Pirates Plunge).

 The Park is Pirate themed due to Wesport House's association with Grace O'Malley.

 The Park was developed by The 11th Marquess of Sligo on the grounds of Westport House.
</item 1>

<item 2>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 2>

<item 3>
Luna Park, Schenectady
Roughly seven decades before the Skycoaster rides that now dot various United States amusement parks, Luna/Rexford Park featured an aerial swing ride.
</item 3>

<item 4>
Christchurch Adventure Park
Christchurch Adventure Park is an Adventure park in the Port Hills of Christchurch, New Zealand.

 Built by the Canadian company Select Evolution, it had its opening function on 16 December 2016.

 The 1500 holders of special passes could ride from 17 December, while the park opened to the public on 21 December.

 According to the developer, the 358 ha park with a 1.8 km chairlift and initially 50 km of downhill tracks is the largest facility of its kind in the Southern
Hemisphere.

 The park closed on 13 February as a precaution due to a nearby fire and two days later, most of the park's tree cover had been destroyed in a large wildfire.

 The lift and ziplines have suffered significant damage, and the park's condition is much worse than the operator initially thought.

 No reopening date has been set, and "it won't be a quick fix".
</item 4>

<item 5>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 5>

<item 6>
Wild West Express Coaster
Wild West Express Coaster is a steel roller coaster operating at Glenwood Caverns Adventure Park in Glenwood Springs, Colorado.

 Wild West Express Coaster opened to the public at Glenwood Caverns Adventure Park on May 25, 2012.
</item 6>

<item 7>
White Water Landing (Dorney Park)
White Water Landing is a Shoot-the-Chutes ride at Dorney Park & Wildwater Kingdom amusement park in Allentown, Pennsylvania.

 It was built in 1992, 10 years after a sister park named Cedar Point Built a defunct flume using the same name: White Water Landing (Cedar Point).

 This ride is identical to Snake River Falls at Cedar Point in Sandusky, Ohio.

 It is located in the upper side near Hydra the Revenge.
</item 7>

<item 8>
Glenwood Caverns Adventure Park
Glenwood Caverns Adventure Park is an adventure park located above Glenwood Springs, Colorado, about 160 miles west of Denver.

 Prior to 2003, only cave tours were available until a major expansion took place.

 The park is unique because it sits at an altitude of 7,100 feet on a mountain above Glenwood.

 Today, the park features numerous attractions in addition to the cave tours.
</item 8>

<item 9>
Deering Oaks: 55-acre historic Portland park with sports, pond, playground, farmers market. Berkeley, Denver: growing Northwest Denver neighborhood with two
lake parks and recreation center. Preston, Texas: Republic-era ghost town; cemetery remains. Scusset Beach: Cape Cod Canal beach, campground, jetty, WWII
fortifications. Crystal Lake: 455-acre New Hampshire lake feeding Suncook River. Spring creeks: clear, cold, steady spring-fed rivers. Squire Creek: Louisiana
private
</item 9>

<item 10>
Register Cliff
Register Cliff is a sandstone cliff and featured key navigational landmark prominently listed in the 19th century guidebooks about the Oregon Trail, and a place
where many emigrants chiseled the names of their families on the soft stones of the cliff it was one of the key checkpoint landmarks for parties heading west
along the Platte River valley west of Fort John, Wyoming which allowed travelers to verify they were on the correct path up to South Pass and not moving into
impassable mountain terrainsgeographically, it is on the eastern ascent of the Continental divide leading upward out of the great plains in the east of the U.S.
state of Wyoming.

 It is notable as a historic landmark for 'registering' hundreds of emigrants on the Oregon Trail (thus also the other northern Emigrant Trails that split off
farther west such as the California Trail and Mormon Trail) who came to follow custom and inscribed their names on its rocks during the western migrations of
the 19th century.

 An estimated 500,000 emigrants used these trails from 1843–1869, with up to one-tenth dying along the way, usually due to disease.
</item 10>
=== END ITEM id=fb1aabfa279d0448 ===

=== ITEM id=dd9c43f590d78d80 ===
Gold sentences and facts:
Sentence 0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great Britain and is dedicated to Saint Chad of Mercia.
  Fact 0.0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham.
  Fact 0.1: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the province of the Catholic Church in Great Britain.
  Fact 0.2: The Metropolitan Cathedral Church and Basilica of Saint Chad is dedicated to Saint Chad of Mercia.
Sentence 1:  Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English Reformation and raised to cathedral status in 1852.
  Fact 1.0: St Chad's was built by Augustus Welby Pugin.
  Fact 1.1: St Chad's was substantially complete by 1841.
  Fact 1.2: St Chad's is one of the first four Catholic churches constructed after the English Reformation.
  Fact 1.3: St Chad's was raised to cathedral status in 1852.
Sentence 2: Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.
  Fact 2.0: Chelmsford Cathedral is in the city of Chelmsford, Essex, England.
  Fact 2.1: Chelmsford Cathedral is dedicated to St Mary the Virgin.
  Fact 2.2: Chelmsford Cathedral is dedicated to St Peter.
  Fact 2.3: Chelmsford Cathedral is dedicated to St Cedd.
Sentence 3:  It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
  Fact 3.0: Chelmsford Cathedral became a cathedral when the Anglican Diocese of Chelmsford was created in 1914.
  Fact 3.1: Chelmsford Cathedral is the seat of the Bishop of Chelmsford.

Retrieved items:
<item 1>
St Philip's Cathedral, Birmingham
The Cathedral Church of Saint Philip is the Church of England cathedral and the seat of the Bishop of Birmingham.

 Built as a parish church and consecrated in 1715, St Philip's became the cathedral of the newly formed Diocese of Birmingham in 1905.

 St Philip's was built in the early 18th century in the Baroque style by Thomas Archer and is located on Colmore Row, Birmingham, England.

 The cathedral is a Grade I listed building.

 St Philip's is the third smallest cathedral in England after Derby and Chelmsford.
</item 1>

<item 2>
St Chad's Cathedral, Birmingham
The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great
Britain and is dedicated to Saint Chad of Mercia.

 Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English
Reformation and raised to cathedral status in 1852.

 It is one of only four minor basilicas in England (the others being Downside Abbey, the National Shrine of Our Lady at Walsingham and Corpus Christi Priory,
this last now disused).

 St Chad's is a Grade II* listed building.

 The cathedral is located in a public greenspace near St Chad's Queensway, in central Birmingham.

 s of 2014 the Archbishop was Bernard Longley and the Dean Canon Gerry Breen.
</item 2>

<item 3>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 3>

<item 4>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 4>

<item 5>
St Chad's College, Durham
St Chad's College is a recognised (independent) college of Durham University in England, founded in 1904 as an Anglican hall for the training of Church of
England clergy.

 The main part of the college is located on the Bailey, occupying nine historic buildings at the east end of Durham Cathedral.

 It neighbours Hatfield College to its north, while St John's College and St Cuthbert's Society are to its south.

 The college is named after St Chad of Mercia, a 7th-century bishop.
</item 5>

<item 6>
Dean of Chelmsford
The Dean of Chelmsford is the head ("primus inter pares" – first among equals) and chair of the Cathedral Chapter, the governing body of Chelmsford Cathedral,
the Cathedral Church of St Mary the Virgin, St Peter and St Cedd.

 Before 2000 the post was designated as a provost, which was then the equivalent of a dean at most English cathedrals.

 The cathedral is the mother church of the Diocese of Chelmsford and seat of the Bishop of Chelmsford.

 The Dean of Chelmsford is also responsible for the Chapel of St Peter-on-the-Wall at Bradwell-on-Sea, founded by St Cedd, among the oldest church buildings in
regular use in England.
</item 6>

<item 7>
Lichfield Cathedral
Lichfield Cathedral is situated in Lichfield, Staffordshire, England.

 It is the only medieval English cathedral with three spires.

 The Diocese of Lichfield covers all of Staffordshire, much of Shropshire and part of the Black Country and West Midlands.

 The 99th and current Bishop of Lichfield is Michael Ipgrave who was appointed on 10 June 2016.
</item 7>

<item 8>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
</item 8>

<item 9>
Bishop's House, Birmingham
The Bishop's House in Birmingham, England was designed by Augustus Pugin as the residence of Thomas Walsh, the first Roman Catholic Bishop of Birmingham.

 It was situated opposite St Chad's Cathedral, on the corner of Bath Street and Weaman Street in Birmingham City Centre.
</item 9>

<item 10>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 10>
=== END ITEM id=dd9c43f590d78d80 ===

=== ITEM id=29147bdc1d035828 ===
Gold sentences and facts:
Sentence 0:  The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI Studios in London, which was soon renamed Abbey Road Studios.
  Fact 0.0: The title and cover are an homage to the Beatles album Abbey Road.
  Fact 0.1: 926 East McLemore Avenue is the address of the Stax Studios in Memphis.
  Fact 0.2: Abbey Road was the address of EMI Studios in London.
  Fact 0.3: EMI Studios in London was soon renamed Abbey Road Studios.
Sentence 1:  This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
  Fact 1.0: This road is best known for the Abbey Road Studios.
  Fact 1.1: This road is best known for the 1969 album Abbey Road by The Beatles.

Retrieved items:
<item 1>
Abbey Road, London
Abbey Road is a thoroughfare in the borough of Camden and the City of Westminster in London, running roughly northwest to southeast through St. John's Wood,
near Lord's Cricket Ground.

 It is part of the B507 road.

 This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
</item 1>

<item 2>
City of Westminster Magistrates' Court
The City of Westminster Magistrates' Court was a magistrates' court located at 70 Horseferry Road, in the City of Westminster, London.

 It was originally called Horseferry Road Magistrates' Court, after the road in which it was sited.

 However, it was renamed in July 2006 following the closure of Bow Street Magistrates' Court.

 It served as the court where the Chief Magistrate of England and Wales sat, and all extradition and terrorism-related cases passed through the court.

 The court closed permanently on 22 September 2011, and was replaced on 27 September 2011 with Westminster Magistrates' Court, built on the site of Marylebone
Magistrates' Court at 181 Marylebone Road.
</item 2>

<item 3>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 3>

<item 4>
Temple Bar, London
The term "Temple Bar" strictly refers to a notional bar or barrier across the route, but is commonly used to refer to the 17th-century ornamental Baroque arched
gateway designed by Christopher Wren which spanned the road until its removal in 1878.

 Wren's arch was preserved and was re-erected in 2004 in the City, in Paternoster Square next to St Paul's Cathedral.
</item 4>

<item 5>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 5>

<item 6>
Westminster tube station
Westminster is a London Underground station in the City of Westminster.

 It is served by the Circle, District and Jubilee lines.

 On the Circle and District lines, the station is between St. James's Park and Embankment, and on the Jubilee line it is between Green Park and Waterloo.

 It is in Travelcard Zone 1.

 The station is located at the corner of Bridge Street and Victoria Embankment and is close to the Houses of Parliament, Westminster Abbey, Parliament Square,
Whitehall, Westminster Bridge, and the London Eye.

 Also close by are Downing Street, the Cenotaph, Westminster Millennium Pier, the Treasury, the Foreign and Commonwealth Office, and the Supreme Court.
</item 6>

<item 7>
Temple Bar, London
Temple Bar was the principal ceremonial entrance to the City of London on its western side from the City of Westminster.

 It is situated on the historic royal ceremonial route from the Tower of London to the Palace of Westminster, the two chief residences of the mediaeval English
monarchs, and from the Palace of Westminster to St Paul's Cathedral.

 The road east of Temple Bar and within the City is Fleet Street, the road to the west, in Westminster, is The Strand.

 At Temple Bar the Corporation of the City of London formerly erected a barrier to regulate trade into the City.

 The 19th century Royal Courts of Justice are located next to it on its north side, having been moved from Westminster Hall.

 To its south is the Temple Church and the Inner Temple and Middle Temple Inns of Court.

 As the most important entrance to the City of London from Westminster, it was formerly long the custom for the monarch to halt at Temple Bar before entering
the City of London, in order for the Lord Mayor to offer the Corporation's pearl-encrusted Sword of State as a token of loyalty.
</item 7>

<item 8>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 8>

<item 9>
Westminster
Westminster ( ) is an area of central London within the City of Westminster, part of the West End, on the north bank of the River Thames.

 Westminster's concentration of visitor attractions and historic landmarks, one of the highest in London, includes the Palace of Westminster, Buckingham Palace,
Westminster Abbey and Westminster Cathedral.
</item 9>

<item 10>
Westminster St Margaret and St John
St Margaret was an ancient parish in the City and Liberty of Westminster and the county of Middlesex.

 It included the core of modern Westminster, including the Palace of Westminster and the area around, but not including Westminster Abbey.

 In 1727 it was divided into St Margaret's and St John's, to coincide with the building of the Church of St John the Evangelist, constructed by the Commission
for Building Fifty New Churches in Smith Square to meet the demands of the growing population, but there continued to be a single vestry for the parishes of St
Margaret and St John.

 This was reformed in 1855 by the Metropolis Management Act, and the two parishes formed the Westminster District until 1887.

 In 1889 St Margaret and St John became part of the County of London.

 The vestry was abolished in 1900, to be replaced by Westminster City Council, but St Margaret and St John continued to have a nominal existence until 1922.
</item 10>
=== END ITEM id=29147bdc1d035828 ===

=== ITEM id=46aaac3a261ffa4b ===
Gold sentences and facts:
Sentence 0: Andrew and Jeremy Get Married is a 2004 British documentary film written and directed by Don Boyd for the BBC.
  Fact 0.0: Andrew and Jeremy Get Married is a 2004 film.
  Fact 0.1: Andrew and Jeremy Get Married is a British documentary film.
  Fact 0.2: Andrew and Jeremy Get Married was written by Don Boyd.
  Fact 0.3: Andrew and Jeremy Get Married was directed by Don Boyd.
  Fact 0.4: Andrew and Jeremy Get Married was made for the BBC.
Sentence 1: The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.
  Fact 1.0: The Last Lions is a 2011 film.
  Fact 1.1: The Last Lions is an African nature documentary film.
  Fact 1.2: The Last Lions is by the National Geographic Society.
  Fact 1.3: The Last Lions was videotaped by Dereck and Beverly Joubert.
  Fact 1.4: The Last Lions was directed by Dereck and Beverly Joubert.

Retrieved items:
<item 1>
Two Brides and a Baby
Two Brides and a Baby is a 2011 Nigerian romantic drama film directed by Teco Benson, starring Keira Hewatch, Kalu Ikeagwu, OC Ukeje, Chelsea Eze, Stella
Damasus-Aboderin and Okey Uzoeshi.

 It premiered on November 17, 2011.

 It received awards and nominations at Africa Movie Academy Awards, Best of Nollywood Awards and Africa Magic Viewers Choice Awards.
</item 1>

<item 2>
Curtain Up
Curtain Up is a 1952 British film directed by Ralph Smart, written by Jack Davies and Michael Pertwee.

 It is based on the play "On Monday Next" by Philip King.
</item 2>

<item 3>
Get Married 3
Get Married 3 is an Indonesian romantic comedy directed by Monty Tiwa and released in 2011.

 A sequel to Hanung Bramantyo's "Get Married" and "Get Married 2", it stars Nirina Zubir and Fedi Nuril as a married couple attempting to raise their triplets
while under intense pressure from their family and friends.

 The film was a commercial success and received favourable reviews in "The Jakarta Post" and "Suara Karya".
</item 3>

<item 4>
Andrew and Jeremy Get Married
Andrew and Jeremy Get Married is a 2004 British documentary film written and directed by Don Boyd for the BBC.

 It tells the story of two Englishmen, Andrew Thomas and Jeremy Trafford, as they plan for their commitment ceremony.

 Originally commissioned for the "BBC Storyville" series, the film premiered at the 2004 Toronto International Film Festival.
</item 4>

<item 5>
Prick Up Your Ears
Prick Up Your Ears is a 1987 British film, directed by Stephen Frears, about the playwright Joe Orton and his lover Kenneth Halliwell.

 The screenplay was written by Alan Bennett, based on the biography by John Lahr.

 The film stars Gary Oldman as Orton, Alfred Molina as Halliwell, Wallace Shawn as Lahr and Vanessa Redgrave as Peggy Ramsay.
</item 5>

<item 6>
The Last Lions
The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.

 It was shot at the Okavango Delta in Botswana.

 The film premiered at the Palm Springs International Film Festival in January 2011 and was released in select theaters on February 18, 2011.

 The film follows in the tradition of other National Geographic big cat films, such as "India: Land of the Tiger" and "Eye of the Leopard".
</item 6>

<item 7>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 7>

<item 8>
Foxtrot (2017 film)
Foxtrot (Hebrew: פוֹקְסטְרוֹט‎ ‎ ) is a 2017 Israeli drama film directed by Samuel Maoz.

 It was screened in the competition section of the 74th Venice International Film Festival where it won the Grand Jury Prize Silver Lion.

 It was also screened in the Special Presentations section at the 2017 Toronto International Film Festival.

 It won the Ophir Award for Best Film, therefore becoming the Israeli entry for the Best Foreign Language Film at the 90th Academy Awards.
</item 8>

<item 9>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 9>

<item 10>
Out of Control (2002 film)
Out of Control is a 2002 British television film written and directed for BBC by Dominic Savage.

 It stars Tamzin Outhwaite as Shelley Richards, an impoverished single mother whose son, Dean (Danny Young), is involved in a string of crimes that lead to his
incarceration in a young offenders institute (YOI).

 David Morrissey plays prison warden Mike, who tries to keep Dean out of trouble but has difficultly watching him all the time.

 After continuous bullying at the YOI, Dean is no longer able to cope and makes a fatal decision.

 It is the third in a loose trilogy of films by Savage about social deprivation, following "Nice Girl" (2000) and "When I Was Twelve" (2001).
</item 10>
=== END ITEM id=46aaac3a261ffa4b ===

=== ITEM id=9f0fc5616f3458ff ===
Gold sentences and facts:
Sentence 0: Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct "Final Destination 5" six years later, and filmed in the IMAX 3D format.
  Fact 0.0: Aliens of the Deep is a 2005 documentary film.
  Fact 0.1: Aliens of the Deep was directed in part by James Cameron.
  Fact 0.2: James Cameron directed Aliens of the Deep alongside fellow cameraman and friend Steven Quale.
  Fact 0.3: Steven Quale would go on to direct Final Destination 5 six years later.
  Fact 0.4: Aliens of the Deep was filmed in the IMAX 3D format.
Sentence 1:  Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of the planet's more unusual forms of life.
  Fact 1.0: James Cameron teams with NASA scientists to explore the Mid-Ocean Ridges.
  Fact 1.1: The Mid-Ocean Ridges are submerged chains of mountains.
  Fact 1.2: The Mid-Ocean Ridges are in the Atlantic and Pacific oceans.
  Fact 1.3: The Mid-Ocean Ridges are home to some of the planet's more unusual forms of life.
Sentence 2: Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.
  Fact 2.0: Poetry in Motion is a 1982 Canadian documentary film.
  Fact 2.1: Poetry in Motion was directed by Ron Mann.
  Fact 2.2: Poetry in Motion features contemporary North American poetry.
  Fact 2.3: Poetry in Motion features contemporary North American music.
Sentence 3:  Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.
  Fact 3.0: Poetry in Motion features some Black Mountain poets.
  Fact 3.1: Poetry in Motion features some Beat poets.
  Fact 3.2: Poetry in Motion features some minimalist poets.
  Fact 3.3: Poetry in Motion features some avant-garde poets.
Sentence 4:  It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.
  Fact 4.0: Poetry in Motion was released in theaters.
  Fact 4.1: Poetry in Motion was later distributed on VHS.
  Fact 4.2: Poetry in Motion was later distributed on LaserDisc.
  Fact 4.3: Poetry in Motion was later distributed on DVD.
Sentence 5:  An extended CD-ROM version was also released.
  Fact 5.0: An extended CD-ROM version of Poetry in Motion was also released.

Retrieved items:
<item 1>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 1>

<item 2>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 2>

<item 3>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 3>

<item 4>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 4>

<item 5>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 5>

<item 6>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 6>

<item 7>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 7>

<item 8>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 8>

<item 9>
The Black Hole
The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.

 The film stars Maximilian Schell, Robert Forster, Joseph Bottoms, Yvette Mimieux, Anthony Perkins, and Ernest Borgnine, while the voices of the main robot
characters are provided by Roddy McDowall and Slim Pickens (both unbilled).

 The music for the film was composed by John Barry.

 It was the first film from Walt Disney Productions to receive a PG rating.

 The film was released on December 18, 1979 in the United Kingdom and on December 21, 1979 in the United States.
</item 9>

<item 10>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 10>
=== END ITEM id=9f0fc5616f3458ff ===
