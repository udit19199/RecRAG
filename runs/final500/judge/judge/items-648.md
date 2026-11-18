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

=== ITEM id=9bb5362a619d2160 ===
Gold sentences and facts:
Sentence 0:  It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
  Fact 0.0: The book is a companion book to 'A Wrinkle in Time'.
  Fact 0.1: The book is part of the Time Quintet.
Sentence 1: The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
  Fact 1.0: The Time Quintet is a fantasy/science fiction series.
  Fact 1.1: The Time Quintet is a series of five young adult novels.
  Fact 1.2: The Time Quintet was written by Madeleine L'Engle.

Retrieved items:
<item 1>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 1>

<item 2>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 2>

<item 3>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 3>

<item 4>
Dust in the Wind (film)
Dust in the Wind is a 1986 film by Taiwanese filmmaker Hou Hsiao-hsien.

 It is based on co-screenwriter Wu Nien-jen's own experiences, and is the final part of Hou's coming-of-age trilogy, with the others being "A Summer at
Grandpa's" (1984) and "The Time to Live and the Time to Die" (1985).
</item 4>

<item 5>
The Four Winds of Love
The Four Winds of Love is the overall title for a series of six novels written by Compton Mackenzie, "The East Wind of Love" (1937), "The South Wind of Love"
(1938), "The West Wind of Love" (1940), "West to North" (1942), "The North Wind of Love, Book 1" (1944) and "The North Wind of Love, Book 2" (1945), which taken
together constitute a major fictional chronicle of the first forty years of the twentieth century.

 The main protagonist of the hexalogy is the semi-autobiographical character of John Ogilvie.
</item 5>

<item 6>
Time Quintet
The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
</item 6>

<item 7>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 7>

<item 8>
The Frost King
"The Frost King" was a short story about King Jack Frost written by 11-year-old Helen Keller.

 Keller's teacher, Anne Sullivan, had mentioned that the autumn leaves were "painted ruby, emerald, gold, crimson, and brown," and Keller, by her own account,
imagined fairies doing the work.

 Keller wrote a story about how a cask of jewels, being transported by fairy servants, had melted in the sun and covered the leaves.

 As a birthday gift, Keller sent the story to Michael Anagnos, head of the Perkins School for the Blind, who published the story in "The Mentor," the Perkins
alumni magazine.

 It was picked up by "The Goodson Gazette", a journal on deaf-blind education, based in Virginia.
</item 8>

<item 9>
Gate of Ivory, Gate of Horn
Gate of Ivory, Gate of Horn is a fantasy novel by British author Robert Holdstock.

 It was originally published in the United States in 1997 (and in the United Kingdom under the title "Gate of Ivory" in 1998.)

 The story is a prequel to "Mythago Wood" and explores Christian Huxley's quest into Ryhope Wood and the apparent suicide of his mother, Jennifer Huxley.

 The title of the book refers to the gates of horn and ivory described in both Homer's "Odyssey" and Virgil's "Aeneid".
</item 9>

<item 10>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 10>
=== END ITEM id=9bb5362a619d2160 ===

=== ITEM id=f11dbd57d2e332ca ===
Gold sentences and facts:
Sentence 0:  The Gaels, led by sixth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic Conference (MAAC).
  Fact 0.0: The Gaels were led by sixth year head coach Tim Cluess.
  Fact 0.1: The Gaels played their home games at the Hynes Athletic Center.
  Fact 0.2: The Gaels were members of the Metro Atlantic Athletic Conference (MAAC).
Sentence 1:  The Cyclones were coached by Steve Prohm, who was in his 1st season.
  Fact 1.0: The Cyclones were coached by Steve Prohm.
  Fact 1.1: Steve Prohm was in his 1st season.

Retrieved items:
<item 1>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 1>

<item 2>
Bob Hurley
Robert Emmet Hurley, Sr. (born July 31, 1947) is a basketball coach at St. Anthony High School in Jersey City, New Jersey.

 Hurley has amassed 28 state championships and more than 1000 wins in 39 years as a coach.

 On February 2, 2011, Hurley became the tenth coach in high school history to win 1000 games.

 He is featured in the documentary "The Street Stops Here".

 Five of his teams have gone undefeated.

 On April 5, 2010, he was announced as the only coach to be inducted into the Naismith Memorial Basketball Hall of Fame that year and only the third high school
coach in history to be so honored; he was formally inducted on August 13 of that year.

 Hurley is the father of Bobby Hurley, a former All-American point guard at Duke and the head basketball coach at Arizona State, and Dan Hurley, who was hired
in February 2012 to coach the University of Rhode Island after two years of coaching at Wagner College and nine years coaching at Newark's Saint Benedict's
Preparatory School, also one of the top high school programs in the nation.
</item 2>

<item 3>
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 3>

<item 4>
Richard F. Wilson
Richard F. Wilson was selected to be Illinois Wesleyan University's 18th President on April 6, 2004 and took office on July 1, 2004.

 He was inaugurated on April 9, 2005.

 He had spent the previous 26 years at the University of Illinois at Urbana-Champaign, leaving as associate chancellor for development and vice president of the
University of Illinois Foundation.

 Mr. Wilson is a Ph.D. graduate of the University of Michigan.

 Wilson announced his intended retirement from Illinois Wesleyan, to take place around June 2015 or whenever a successor is named and transitioned.
</item 4>

<item 5>
2015–16 Iowa State Cyclones men's basketball team
The 2015–16 Iowa State Cyclones men's basketball team represents Iowa State University during the 2015–16 NCAA Division I men's basketball season.

 The Cyclones were coached by Steve Prohm, who was in his 1st season.

 They played their home games at Hilton Coliseum in Ames, Iowa and competed in the Big 12 Conference.
</item 5>

<item 6>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 6>

<item 7>
Nate Oats
Nathanael "Nate" Oats (born October 13, 1974) is an American basketball coach.

 He is currently the head basketball coach at the State University of New York at Buffalo, following two seasons as an assistant coach under Bobby Hurley.

 Oats was named head coach on April 11, 2015 after Hurley was hired by Arizona State as head coach.

 In only his first season as head coach, he took the Buffalo Bulls to their second straight NCAA Tournament bid after winning the Mid-American Conference
Championship.
</item 7>

<item 8>
Sylvester Croom
Sylvester Croom Jr. (born September 25, 1954) is an American football coach.

 He is currently the running backs coach for the NFL's Tennessee Titans.

 He was the head coach at Mississippi State University from 2004 to 2008, and the first African American head football coach in the Southeastern Conference.

 His father, Sylvester Croom, Sr., was himself an All-American football player at Alabama A&M, later the team chaplain at the University of Alabama, and has
been recognized by that school as one of the state's 40 pioneers of civil rights.

 Since his time at Mississippi State, Croom, Jr. has served as running backs coach for three teams in the National Football League.
</item 8>

<item 9>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 9>

<item 10>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 10>
=== END ITEM id=f11dbd57d2e332ca ===

=== ITEM id=5b2e0bbb116662db ===
Gold sentences and facts:
Sentence 0:  He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
  Fact 0.0: The person referred to as 'He' has had a cameo acting role as a tailor in The Wolf of Wall Street.
  Fact 0.1: The person referred to as 'He' has had a cameo acting role as a tailor in The Good Shepherd.
Sentence 1: The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir of the same name by Jordan Belfort.
  Fact 1.0: The Wolf of Wall Street is a 2013 film.
  Fact 1.1: The Wolf of Wall Street is American.
  Fact 1.2: The Wolf of Wall Street is a biographical black comedy crime film.
  Fact 1.3: The Wolf of Wall Street was directed by Martin Scorsese.
  Fact 1.4: The Wolf of Wall Street was written by Terence Winter.
  Fact 1.5: The Wolf of Wall Street is based on the memoir of the same name by Jordan Belfort.

Retrieved items:
<item 1>
The Wolf of Wall Street (2013 film)
The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir
of the same name by Jordan Belfort.

 It recounts Belfort's perspective on his career as a stockbroker in New York City and how his firm Stratton Oakmont engaged in rampant corruption and fraud on
Wall Street that ultimately led to his downfall.

 Leonardo DiCaprio (who was also a producer) stars as Belfort, with Jonah Hill as his business partner and friend Donnie Azoff, Margot Robbie as his wife Naomi
Lapaglia and Kyle Chandler as Patrick Denham, the FBI agent who tries to bring him down.

 Matthew McConaughey, Rob Reiner, Jon Favreau, Joanna Lumley and Jean Dujardin also star.
</item 1>

<item 2>
Leonard Logsdail
Leonard Logsdail (born September 11, 1950 in London, England) is a bespoke tailor, based in Manhattan, New York, specializing in men's suits.

 He is described as one of the finest bespoke tailors in the men’s suit business.

 All of Logsdail's suits are crafted and perfected on-site in his New York City location.

 He is credited for making Savile Row tailoring a local option in New York.

 Logsdail is known for creating high-end suits, including lining jackets with Hermes silk scarves.

 CNBC talk show host, economist and fashion icon Larry Kudlow has his suits made by Leonard Logsdail.

 Logsdail has created suits for award-winning films, and is recognized as one of cinema's most sought-after tailors.

 Logsdail has collaborated with esteemed film directors Steven Spielberg, Robert de Niro, Oliver Stone, Ridley Scott, and Martin Scorcese.

 He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
</item 2>

<item 3>
The Wolf of Wall Street (2013 film)
The film marks the director's fifth collaboration with DiCaprio, after "Gangs of New York" (2002), "The Aviator" (2004), "The Departed" (2006) and "Shutter
Island" (2010), as well as his second collaboration with Winter after the television series "Boardwalk Empire" (2010–14).
</item 3>

<item 4>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 4>

<item 5>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 5>

<item 6>
A Walk in the Woods (film)
A Walk in the Woods is a 2015 American biographical comedy-drama film directed by Ken Kwapis, based on the 1998 book/memoir of the same name by Bill Bryson.

 The film stars Robert Redford, Nick Nolte and Emma Thompson.

 The film was released on September 2, 2015, by Broad Green Pictures.
</item 6>

<item 7>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 7>

<item 8>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
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
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 10>
=== END ITEM id=5b2e0bbb116662db ===

=== ITEM id=5bde4d28943aae16 ===
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
Somebody Up There Likes Me (1956 film)
Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.

 Joseph Ruttenberg was awarded a 1956 Oscar in the category of Best Cinematography (Black and White).

 The film also won the Oscar for Best Art Direction (Cedric Gibbons, Malcolm F. Brown, Edwin B. Willis, F. Keogh Gleason).

 It was directed by Robert Wise.
</item 2>

<item 3>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 3>

<item 4>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 4>

<item 5>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 5>

<item 6>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 6>

<item 7>
Rocky V
Rocky V is a 1990 American sports drama film.

 It is the fifth film in the "Rocky" series, written by and starring Sylvester Stallone, and co-starring Talia Shire, Stallone's real-life son Sage, and
real-life boxer Tommy Morrison, with Morrison in the role of Tommy Gunn, a talented yet raw boxer.

 Sage played Robert Balboa, whose relationship with his famous father is explored.

 After Stallone directed the second through fourth films in the series, "Rocky V" saw the return of John G. Avildsen, whose direction of "Rocky" won him an
Academy Award for Best Directing.
</item 7>

<item 8>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 8>

<item 9>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 9>

<item 10>
The Man with the Golden Arm
The Man with the Golden Arm is a 1955 American drama film with elements of film noir, based on the novel of the same name by Nelson Algren, which tells the
story of a drug addict who gets clean while in prison, but struggles to stay that way in the outside world.

 Although the addictive drug is never identified in the film, according to the American Film Institute "most contemporary and modern sources assume that it is
heroin", in contrast to Algren's book which named the drug as morphine.

 The film stars Frank Sinatra, Eleanor Parker, Kim Novak, Arnold Stang and Darren McGavin.

 It was adapted for the screen by Walter Newman, Lewis Meltzer and Ben Hecht (uncredited), and directed by Otto Preminger.

 The film's initial release sparked controversy due to its serious, in-depth treatment of the then-taboo subject of drug addiction.
</item 10>
=== END ITEM id=5bde4d28943aae16 ===

=== ITEM id=06af01fcb7e0d75e ===
Gold sentences and facts:
Sentence 0:  Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
  Fact 0.0: Cacareco's candidacy inspired the Rhinoceros Party of Canada.
  Fact 0.1: The Rhinoceros Party of Canada was nominally led by the rhinoceros Cornelius the First.
Sentence 1: The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.
  Fact 1.0: The Rhinoceros Party's French name is 'Parti Rhinoceros'.
  Fact 1.1: The Rhinoceros Party was a registered political party in Canada.
  Fact 1.2: The Rhinoceros Party was a registered political party in Canada from the 1960s to the 1990s.

Retrieved items:
<item 1>
Rhinoceros Party of Canada (1963–93)
The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.

 Operating within the tradition of political satire, the Rhinoceros Party's basic credo, their so-called primal promise, was "a promise to keep none of our
promises".

 They then promised outlandishly impossible schemes designed to amuse and entertain the voting public.
</item 1>

<item 2>
Aboriginal Peoples Party of Canada
The Aboriginal Peoples Party of Canada (APP) is a Canadian political party that was founded in 2005.

 The party was conceived by University of Lethbridge student Myron Wolf Child.

 It held its founding meeting on August 21, 2005, in St. Albert, Alberta.

 The APP was headed by interim leader Bill Montour, a former Chief of the Six Nations Reserve in Ontario.

 In October 2005, the party had 122 confirmed members, just less than half the number needed to register as an official party in Canada.

 In late October 2005, the APP sought to unite with the First Peoples National Party of Canada which also had fewer than the number of confirmed members needed
to become a registered political party.

 The First Peoples National Party of Canada became an eligible political party on December 6, 2005.

 Whether or not this was accomplished through a merger with the APP is unclear.

 If this is the case, the APP no longer exists, but, as one of the stipulations of a possible merger was that a name for the new party would be determined at
the party's first convention, it is possible that the name Aboriginal Peoples Party of Canada will come into usage again.
</item 2>

<item 3>
National Advancement Party of Canada
The National Advancement Party of Canada (French: "Parti de l'Essor National du Canada" ) is a minor federal political party in Canada.

 The party describes itself as for "advancement of the Canadian people and Canada".

 The party was formed in 2014 and registered with Elections Canada in 2015, initially under the name Democratic Advancement Party of Canada.

 The party is headquartered in Calgary, Alberta and led by writer Stephen Garvey.
</item 3>

<item 4>
Revolutionary Communist Party of Canada
The Revolutionary Communist Party of Canada, or RCP Canada, is a revolutionary communist party oriented around Marxism-Leninism-Maoism.

 The creation of the organizational stage of the Party was adopted at what was called the "Revolutionary Communist Conference", which was held in Montreal,
Quebec in November 2000 by activists and former members of the labour union movements and youth organizations of the 1960s, 1970s and 1980s, who felt that the
revolutionary situation in Canada warranted the creation of a Party dedicated to a communist revolution.

 The party was initially called the Revolutionary Communist Party (Organizing Committees) (RCP(OC)).

 At this conference, participants adopted the Party's first Draft Programme.

 Because of the location of the conference, the majority of the founding members were French-speaking Quebecers, and the Party began an extensive effort to
reach out to the rest of Canada, starting with the Canadian Revolutionary Congress held in November 2006 in Toronto.

 It is not recognized as a political party by Elections Canada, because the party rejects what it calls the "bourgeois electoral system" and doesn't seek
recognition by the State.
</item 4>

<item 5>
The New Canada
The New Canada is a Canadian political literature book written by Reform Party of Canada founder and leader Preston Manning and published by Macmillan Canada.

 The book explains the personal, religious, and political life of Preston Manning and explains the roots and beliefs of the Reform Party.

 At the time of its publishing in 1992, Reform had become a popular populist conservative party in Western Canada after the mainstream Progressive Conservative
Party of Canada was collapsing in support and in 1991 decided to expand eastward into Ontario and the Maritimes.

 One year later the PC party collapsed in the 1993 federal election, allowing the Reform Party to make political history in Canada, deplacing the PCs as the
dominant conservative party in Canada.

 Reform, later renamed the Canadian Alliance, merged with the PC Party in 2003, to form a united right-wing alternative to the governing Liberal Party of
Canada, named the Conservative Party of Canada which has dropped many of the populist themes that the Reform Party had.
</item 5>

<item 6>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 6>

<item 7>
Cacareco
Cacareco (1954-1962) was a female black rhinoceros exhibited in Brazilian zoos.

 She became famous as a candidate for the 1958 São Paulo city council elections with the intention of protesting against political corruption.

 Electoral officials did not accept Cacareco's candidacy, but she eventually won 100,000 votes, more than any other party in that same election (which was also
marked by rampant absenteeism).

 Today, the term "voto Cacareco" (Cacareco vote) is commonly used to describe protest votes in Brazil.

 Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
</item 7>

<item 8>
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
</item 8>

<item 9>
Western Canada Miners
The Western Canada Miners (also known as just the Canada Miners) are a professional independent baseball team based in Yuma, Arizona, and representing western
Canada.

 They play in the developmental Arizona Winter League, a short-season instructional winter league affiliated with the North American League and they play their
home games at Desert Sun Stadium in Yuma, as well as the San Diego Surf Dawgs, Saskatchewan Silver Sox, Team Canada and Yuma Scorpions.

 The team uses the logo and uniforms of the now-defunct Mesa Miners team and are owned by Diamond Sports & Entertainment.
</item 9>

<item 10>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 10>
=== END ITEM id=06af01fcb7e0d75e ===

=== ITEM id=ef2a6ee1256867b8 ===
Gold sentences and facts:
Sentence 0:  The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.
  Fact 0.0: The group included Alexander Bogdanov.
  Fact 0.1: The group included Maxim Gorky.
  Fact 0.2: The group included Anatoly Lunacharsky.
  Fact 0.3: The group included Mikhail Pokrovsky.
  Fact 0.4: The group included Grigory Aleksinsky.
  Fact 0.5: The group included Stanislav Volski.
  Fact 0.6: The group included Martyn Liadov.
Sentence 1: Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.
  Fact 1.0: Anatoly Vasilyevich Lunacharsky's Russian name is Анато́лий Васи́льевич Лунача́рский.
  Fact 1.1: Anatoly Vasilyevich Lunacharsky was born on 23 November 1875 (Old Style 11 November 1875).
  Fact 1.2: Anatoly Vasilyevich Lunacharsky died on 26 December 1933.
  Fact 1.3: Anatoly Vasilyevich Lunacharsky was a Russian Marxist revolutionary.
  Fact 1.4: Anatoly Vasilyevich Lunacharsky was the first Soviet People's Commissar of Education.
  Fact 1.5: Anatoly Vasilyevich Lunacharsky was responsible for culture and education as People's Commissar of Education.

Retrieved items:
<item 1>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 1>

<item 2>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 2>

<item 3>
You Fell Victim to a Fateful Struggle
You Fell Victim (Russian: "Вы жертвою пали"; "Vy zhertvoiu pali"), or You Fell Victim to a Fateful Struggle, is a Russian Marxist and revolutionary funeral
march.

 It acted as the funeral dirge of the Russian revolutionary movement, among them the Bolsheviks.
</item 3>

<item 4>
Vpered
Vpered ("Forward" or "Hasten") (1909 - 1912) was an organization emanating from within the Russian Social Democratic Labour Party (Russian Social Democracy or
RSDLP).

 The faction was gathered by Alexander Bogdanov (1873 - 1928) in December 1909.

 The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.

 Although Vpered emerged from the Bolshevik wing of Russian Social Democracy, the group was critical of Lenin.
</item 4>

<item 5>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 5>

<item 6>
Sergey Yustinovich Bagotsky
Sergey Yustinovich Bagotsky (Russian: Серге́й Юстинович Багоцкий ; 15 February 1879 - 15 March 1953) was a Russian Marxist revolutionary and Soviet medicine and
Red Cross diplomat, Soviet head of the Red Cross representative mission to Geneva from 1918 to 1936.
</item 6>

<item 7>
Apollinariya Yakubova
Apollinariya Yakubova (died 1913 or 1917) was a Russian Marxist revolutionary and, with Vladimir Lenin, one of the founders of the League of Struggle for the
Emancipation of the Working Class.
</item 7>

<item 8>
Nikolay Speshnev
Nikolay Alexandrovich Speshnev (Russian: Николай Александрович Спе́шнев; 1821, Kursk - 1882, St. Petersburg) was a 19th-century Russian aristocrat and political
activist, best known for his involvement with the pro-socialist literary discussion group the Petrashevsky Circle.

 He formed a secret revolutionary society from among the members of the circle, which included the young Fyodor Dostoevsky.

 After the government of Tsar Nicholas I arrested the members of the Petrashevsky Circle in 1849, Speshnev was interrogated, threatened with torture, and
eventually sentenced, along with Dostoevsky, Petrashevsky and others, to execution by firing squad.

 The sentence was commuted to hard labour in Siberia, but the prisoners were only informed of this after enduring a mock execution.
</item 8>

<item 9>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 9>

<item 10>
Matvey Skobelev
Matvey Ivanovich Skobelev (Russian: Матве́й Ива́нович Ско́белев ; November 9, 1885, Baku – July 29, 1938, Moscow) was a Russian Marxist revolutionary and
politician.
</item 10>
=== END ITEM id=ef2a6ee1256867b8 ===

=== ITEM id=334e44015d1d8a1d ===
Gold sentences and facts:
Sentence 0: John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.
  Fact 0.0: John Kingsley Orton, known as Joe Orton, was born on 1 January 1933.
  Fact 0.1: Joe Orton died on 9 August 1967.
  Fact 0.2: Joe Orton was an English playwright.
  Fact 0.3: Joe Orton was an English author.
Sentence 1:  His public career was short but prolific, lasting from 1964 until his death three years later.
  Fact 1.0: Joe Orton's public career was short but prolific.
  Fact 1.1: Joe Orton's public career lasted from 1964 until his death.
  Fact 1.2: Joe Orton died three years after his public career began in 1964.
Sentence 2:  During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.
  Fact 2.0: During his brief public career, Joe Orton shocked audiences with his scandalous black comedies.
  Fact 2.1: During his brief public career, Joe Orton outraged audiences with his scandalous black comedies.
  Fact 2.2: During his brief public career, Joe Orton amused audiences with his scandalous black comedies.
Sentence 3:  The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
  Fact 3.0: The adjective 'Ortonesque' is sometimes used to refer to work characterised by a dark yet farcical cynicism.
  Fact 3.1: The cynicism is similarly dark yet farcical.
Sentence 4: Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary modernism, and is considered by many to be among the greatest novelists in the English language.
  Fact 4.0: Henry James, OM, was an American author.
  Fact 4.1: Henry James was born in 1843.
  Fact 4.2: Henry James died in 1916.
  Fact 4.3: Henry James was regarded as a key transitional figure between literary realism and literary modernism.
  Fact 4.4: Henry James is considered by many to be among the greatest novelists in the English language.
Sentence 5:  He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
  Fact 5.0: Henry James was the son of Henry James, Sr.
  Fact 5.1: Henry James was the brother of the renowned philosopher and psychologist William James.
  Fact 5.2: Henry James was the brother of diarist Alice James.

Retrieved items:
<item 1>
The Henry James Review
The Henry James Review is a triannual peer-reviewed academic journal established in 1979 and is the official publication of the Henry James Society.

 It is dedicated to the scholarly, critical, and theoretical study of the American writer Henry James.

 Each issue focuses on a specific theme of interest and seeks to promote understanding and study of James' contributions.

 The journal is published by the Johns Hopkins University Press and the current editor-in-chief is Susan M. Griffin (University of Louisville).
</item 1>

<item 2>
Henry James
Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary
modernism, and is considered by many to be among the greatest novelists in the English language.

 He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
</item 2>

<item 3>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 3>

<item 4>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 4>

<item 5>
What the Butler Saw (play)
What the Butler Saw is a farce written by the English playwright Joe Orton.

 It was premièred at the Queen's Theatre in London on 5 March 1969.

 It was Orton's final play and the second to be performed after his death, following "Funeral Games" in 1968.
</item 5>

<item 6>
George Orwell
Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.

 His work is marked by lucid prose, awareness of social injustice, opposition to totalitarianism, and outspoken support of democratic socialism.
</item 6>

<item 7>
The Ruffian on the Stair
The Ruffian On the Stair is a play by British playwright Joe Orton which was first broadcast on BBC Radio in August 1964.

 It is an unsympathetic yet comedic one-act portrayal of working class England, as played out by a couple and a mysterious young man who toys with their lives.

 It was based on "The Boy Hairdresser", a novel by Orton and his lover Kenneth Halliwell.

 The title and play are based on a few lines from poet and dramatist William Ernest Henley: "Madam Life's a piece in bloom, / Death goes dogging everywhere: /
She's the tenant of the room, / He's the ruffian on the stair."

 "Ruffian" is not as renowned as other works such as "Loot" and "What the Butler Saw", but it is still staged on occasion.
</item 7>

<item 8>
Prick Up Your Ears
Prick Up Your Ears is a 1987 British film, directed by Stephen Frears, about the playwright Joe Orton and his lover Kenneth Halliwell.

 The screenplay was written by Alan Bennett, based on the biography by John Lahr.

 The film stars Gary Oldman as Orton, Alfred Molina as Halliwell, Wallace Shawn as Lahr and Vanessa Redgrave as Peggy Ramsay.
</item 8>

<item 9>
Prick Up Your Ears (play)
Prick Up Your Ears is a play by Simon Bent, based on the life of playwright Joe Orton.

 Produced by Sonia Friedman it opened at the Comedy Theatre in London's West End on 30 September 2009 following previews from 17 September.

 It starred Chris New as Joe Orton and Matt Lucas as Orton's lover and murderer, Kenneth Halliwell.
</item 9>

<item 10>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 10>
=== END ITEM id=334e44015d1d8a1d ===

=== ITEM id=f388cb4160643ad7 ===
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
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 1>

<item 2>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 2>

<item 3>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 3>

<item 4>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 4>

<item 5>
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
</item 5>

<item 6>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
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
</item 8>

<item 9>
Mike Mattison
Mike Mattison is an American musician and vocalist of the Grammy Award-winning blues rock/soul group, the Tedeschi Trucks Band, as well as lead vocalist and
co-founder of the blues rock trio Scrapomatic.

 Mattison's vocal sound has been described as "strong," with an "expressive blues voice".

 As lead vocalist of Scrapomatic, he picked up a nomination for Minnesota Music Awards best male vocalist, and both he and co-founder Paul Olsen were also
nominated for best R&B Group.

 Mattison was previously the lead vocalist of the Grammy Award winning Derek Trucks Band and has been a main songwriter of all three bands.

 He is also an active essayist who publishes on music and poetry.

 Since 2013 Mattison and Ernest Suarez have edited “Hot Rocks: Songs and Verse,” an ongoing feature in Five Points: A Journal of Literature and Art.
</item 9>

<item 10>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 10>
=== END ITEM id=f388cb4160643ad7 ===

=== ITEM id=1b042fbd38be96c2 ===
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
</item 2>

<item 3>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 3>

<item 4>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 4>

<item 5>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 5>

<item 6>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
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
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
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
=== END ITEM id=1b042fbd38be96c2 ===

=== ITEM id=f624083bbb5e9b7f ===
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
Quiet Riot 10
Quiet Riot 10 (also alternatively known as just 10, or Quiet Riot Number 10) is the twelfth studio album by the heavy metal band Quiet Riot, which was released
on June 27, 2014.

 It is their first studio album since 1988's "QR" not to feature longtime and founding vocalist Kevin DuBrow in any newly recorded material, due to his death in
November 2007.

 It is also the band's first album since reuniting in 2010.

 Although a studio album, the final four tracks on "Quiet Riot 10" are live performances taken from some of the band's final shows with DuBrow in 2007.

 Love/Hate vocalist Jizzy Pearl joined the band in November 2013 and performs lead vocals on the six studio tracks.
</item 1>

<item 2>
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
</item 2>

<item 3>
Quiet in Heaven/Song of Liberty
"Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the
majority of their members quit and formed Faith No More.

 The songs have not appeared in any of the band's compilation albums and have not been performed live since Mike Morris left the band.
</item 3>

<item 4>
String Quartet No. 6 (Bartók)
Due to the difficulties of the war, communication between Bartók and Székely was difficult, and the quartet was not premiered until 20 January 1941, when the
Kolisch Quartet, to whom the work is dedicated, gave its premiere at the Town Hall in New York City.
</item 4>

<item 5>
Corcovado (song)
"Corcovado" (known in English as "Quiet Nights of Quiet Stars") is a bossa nova song written by Antônio Carlos Jobim in 1960.

 An English lyric was later written by Gene Lees.

 The Portuguese title refers to the Corcovado mountain in Rio de Janeiro.

 Andy Williams recorded the song with English lyrics, reaching #92 in the "Billboard" Hot 100 and #18 in the Hot Adult Contemporary Tracks chart in 1965.

 Also receiving air-play, contemporaneously with Andy Williams' recording of "Quiet Nights," was Kitty Kallen's version.

 Her album, titled "Quiet Nights," was released by 20th Century-Fox Records in 1964.
</item 5>

<item 6>
Jefferson's Tree of Liberty
On February 1, 2009, more tour dates and venues were announced by the band's manager Michael Gaiman, with additional plans to continue the tour through 2010 and
bring it to Australia and South America.

 David Grisman joined the band for the April 2009 tour dates.

 The band's promotion for the album ended in June 2009 as the band changed their set-list to Jefferson Airplane's Woodstock Festival material and started
touring with the "Heroes of Woodstock" through the end of October.
</item 6>

<item 7>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 7>

<item 8>
Kiss Me Quiet (song)
"Kiss Me Quiet" is a song recorded by Canadian country singer Jess Moskaluke for her second extended play of the same name (2015).

 It was written by Kelly Archer, Corey Crowder, and Emily Shackelton and was produced by Crowder.

 "Kiss Me Quiet" was released August 28, 2015 as the EP's lead single.

 The song was Moskaluke's first solo top 10 on the Canada Country chart.
</item 8>

<item 9>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 9>

<item 10>
The Promised Land (Sagol 59 album)
A live show recording was released digitally in April 2017 as "The Promised Land at The Deanery, Brooklyn NY 3-23-2017".

 Live in Cafe Bialik, Tel Aviv, a soundboard concert recording, was released July 4th, 2017.
</item 10>
=== END ITEM id=f624083bbb5e9b7f ===

=== ITEM id=95fc52fcedce5b4d ===
Gold sentences and facts:
Sentence 0: Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in China and Japan.
  Fact 0.0: Pterostyrax is also called the epaulette tree.
  Fact 0.1: Pterostyrax is a small genus.
  Fact 0.2: Pterostyrax has four species.
  Fact 0.3: Pterostyrax species are deciduous.
  Fact 0.4: Pterostyrax species are large shrubs or small trees.
  Fact 0.5: Pterostyrax is in the family Styracaceae.
  Fact 0.6: Pterostyrax is native to eastern Asia.
  Fact 0.7: Pterostyrax is native to China.
  Fact 0.8: Pterostyrax is native to Japan.
Sentence 1: Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.
  Fact 1.0: Dregea is a genus of vines.
  Fact 1.1: Dregea is in the family Apocynaceae.
  Fact 1.2: Dregea was first described as a genus with this name in 1838.
Sentence 2:  It is native to Africa and southern Asia.
  Fact 2.0: Dregea is native to Africa.
  Fact 2.1: Dregea is native to southern Asia.

Retrieved items:
<item 1>
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 1>

<item 2>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 2>

<item 3>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 3>

<item 4>
Pterostyrax hispidus
Pterostyrax hispidus, the epaulette tree or fragrant epaulette tree, is a species of flowering plant in the family Styracaceae, native to China and Japan.

 Growing to 15 m tall by 12 m broad, it is a substantial, spreading, deciduous shrub with oval leaves up to 20 cm long, and clusters of pure white, fragrant,
pendent flowers in summer.
</item 4>

<item 5>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 5>

<item 6>
Dregea
Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.

 It is native to Africa and southern Asia.
</item 6>

<item 7>
Dregea sinensis
Dregea sinensis is a species of plant in the Apocynaceae family that is native to China.
</item 7>

<item 8>
Rhabdodontidae
Rhabdodontids were herbivorous ornithopod dinosaurs from the Cretaceous Period.

 Rhabdodontids were similar to large, robust "hypsilophodonts", with deep skulls and jaws.

 The family was first proposed by David B. Weishampel and colleagues in 2002.

 The rhabdodontids were defined as "the most recent common ancestor of "Zalmoxes robustus" and "Rhabdodon priscus" and all the descendants of this common
ancestor".

 In 2005, Paul Sereno defined the family as "the most inclusive clade containing "Rhabdodon priscus" but not "Parasaurolophus walkeri"".

 The Rhabdodontidae includes the type genus "Rhabdodon", "Zalmoxes", "Mochlodon" and possibly "Muttaburrasaurus".

 Rhabdodontid fossils have been found in Europe and Australia in formations dating from the Early to Late Cretaceous, 100 to 66 million years ago.

 Ősi "et al."
</item 8>

<item 9>
Dregea volubilis
Dregea volubilis is a species of plant in the Apocynaceae family that is native to India.
</item 9>

<item 10>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 10>
=== END ITEM id=95fc52fcedce5b4d ===
