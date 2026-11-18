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

This file has 6 items.

=== ITEM id=baada87941d2cda2 ===
Gold sentences and facts:
Sentence 0: Drawing is one of the major forms of expression within the visual arts . It is generally concerned with the marking of lines and areas of tone onto paper / other material , where the accurate representation of the visual world is expressed upon a plane surface . Traditional drawings were monochrome , or at least had little colour , while modern colored - pencil drawings may approach or cross a boundary between drawing and painting . In Western terminology , drawing is distinct from painting , even though similar media often are employed in both tasks . Dry media , normally associated with drawing , such as chalk , may be used in pastel paintings . Drawing may be done with a liquid medium , applied with brushes or pens . Similar supports likewise can serve both : painting generally involves the application of liquid paint onto prepared canvas or panels , but sometimes an underdrawing is drawn first on that same support .
  Fact 0.0: Drawing is one of the major forms of expression within the visual arts.
  Fact 0.1: Drawing is generally concerned with the marking of lines and areas of tone onto paper or other material.
  Fact 0.2: In drawing, the accurate representation of the visual world is expressed upon a plane surface.
  Fact 0.3: Traditional drawings were monochrome, or at least had little colour.
  Fact 0.4: Modern colored-pencil drawings may approach or cross a boundary between drawing and painting.
  Fact 0.5: In Western terminology, drawing is distinct from painting.
  Fact 0.6: Drawing and painting can employ similar media even though they are distinct tasks.
  Fact 0.7: Dry media normally associated with drawing, such as chalk, may be used in pastel paintings.
  Fact 0.8: Drawing may be done with a liquid medium applied with brushes or pens.
  Fact 0.9: Similar supports can serve both drawing and painting.
  Fact 0.10: Painting generally involves the application of liquid paint onto prepared canvas or panels.
  Fact 0.11: Sometimes an underdrawing is drawn first on the same support that is later painted.

Retrieved items:
<item 1>
Drawing
Look up drawing in Wiktionary , the free dictionary . Wikiversity has learning resources about Drawing Timeline of Drawing Development in Children On Drawing ,
an essay about the craft of drawing , by artist Norman Nason . Archived from the original on April 25 , 2012 . Art - Drawing http://www.drawingforall.net/
Drawing lessons Drawing Resources Line and Form ( 1900 ) by Walter Crane at Project Gutenberg Leonardo da Vinci : anatomical drawings from the Royal Library ,
Windsor Castle , exhibition catalog fully online as PDF from The Metropolitan Museum of Art ( a great drawing resource ) . Leonardo da Vinci , Master Draftsman
, exhibition catalog fully online as PDF from The Metropolitan Museum of Art ( a great drawing resource ) . Drawing in the Middle Ages A summary of how drawing
was used as part of the artistic process in the Middle Ages . ( hide ) Visual arts Architecture Art Ceramics Craft Drawing Design Painting Photography Public
art Sculpture Site - specific art Street art Computer art Filmmaking Printmaking List of artistic media GND : 4127900 - 1 NDL : 00561487 Retrieved from ``
https://en.wikipedia.org/w/index.
</item 1>

<item 2>
Drawing
Sometimes the artist leaves a section of the image untouched while filling in the remainder . The shape of the area to preserve can be painted with masking
fluid or cut out of a frisket and applied to the drawing surface , protecting the surface from stray marks until the mask is removed . Another method to
preserve a section of the image is to apply a spray - on fixative to the surface . This holds loose material more firmly to the sheet and prevents it from
smearing . However the fixative spray typically uses chemicals that can harm the respiratory system , so it should be employed in a well - ventilated area such
as outdoors . Another technique is subtractive drawing in which the drawing surface is covered with graphite or charcoal and then erased to make the image .
Tone ( edit ) A pencil drawing with hatching and shading Shading is the technique of varying the tonal values on the paper to represent the shade of the
material as well as the placement of the shadows . Careful attention to reflected light , shadows and highlights can result in a very realistic rendition of the
image . Blending uses an implement to soften or spread the original drawing strokes .
</item 2>

<item 3>
Drawing
At this point , drawing was commonly used as a tool for thought and investigation , acting as a study medium whilst artists were preparing for their final
pieces of work . The Renaissance brought about a great sophistication in drawing techniques , enabling artists to represent things more realistically than
before , and revealing an interest in geometry and philosophy . The invention of the first widely available form of photography led to a shift in the hierarchy
of the arts . Photography offered an alternative to drawing as a method for accurately representing visual phenomena , and traditional drawing practice was
given less emphasis as an essential skill for artists , particularly so in Western society . Example of a typical drawing by American artist and notable
draftsman , Jean - Michel Basquiat , Untitled ( Axe / Rene ) 1982 . Basquiat produced over 1500 drawings and pushed the boundaries of drawing in the late 20th -
century Notable artists and draftsmen ( edit ) Drawing became significant as an art form around the late 15th century , with artists and master engravers such
as Albrecht Dürer and Martin Schongauer ( c. 1448 - 1491 ) , the very first Northern engraver known by name .
</item 3>

<item 4>
Drawing
These drawings , known as pictograms , depicted objects and abstract concepts . The sketches and paintings produced by Neolithic times were eventually stylised
and simplified in to symbol systems ( proto - writing ) and eventually into early writing systems . Drawing in manuscripts ( edit ) Before the widespread
availability of paper , 12th - century monks in European monasteries used intricate drawings to prepare illustrated , illuminated manuscripts on vellum and
parchment . Drawing has also been used extensively in the field of science , as a method of discovery , understanding and explanation . Drawing in science (
edit ) In 1616 , astronomer Galileo Galilei explained the changing phases of the moon through his observational telescopic drawings . In 1924 , geophysicist
Alfred Wegener used illustrations to visually demonstrate the origin of the continents . Drawing as artistic expression ( edit ) Drawing is used to express one
's creativity , and therefore has been prominent in the world of art . Throughout much of history , drawing was regarded as the foundation for artistic practice
. Initially , artists used and reused wooden tablets for the production of their drawings . Following the widespread availability of paper in the 14th century ,
the use of drawing in the arts increased .
</item 4>

<item 5>
Drawing
Escher Arshile Gorky George Grosz Paul Klee Oscar Kokoschka Käthe Kollwitz Alfred Kubin André Masson Alphonse Mucha Jules Pascin Pablo Picasso Egon Schiele Jean
- Michel Basquiat Andy Warhol Frank Auerbach 21st : Tracey Emin Banksy Materials ( edit ) The medium is the means by which ink , pigment or color are delivered
onto the drawing surface . Most drawing media are either dry ( e.g. graphite , charcoal , pastels , Conté , silverpoint ) , or use a fluid solvent or carrier (
marker , pen and ink ) . Watercolor pencils can be used dry like ordinary pencils , then moistened with a wet brush to get various painterly effects . Very
rarely , artists have drawn with ( usually decoded ) invisible ink . Metalpoint drawing usually employs either of two metals : silver or lead . More rarely used
are gold , platinum , copper , brass , bronze , and tinpoint . Paper comes in a variety of different sizes and qualities , ranging from newspaper grade up to
high quality and relatively expensive paper sold as individual sheets .
</item 5>

<item 6>
Drawing
Drawing - wikipedia Drawing Jump to : navigation , search For other uses , see Drawing ( disambiguation ) . Pen and wash lion by Rembrandt in the Louvre Drawing
is a form of visual art in which a person uses various drawing instruments to mark paper or another two - dimensional medium . Instruments include graphite
pencils , pen and ink , inked brushes , wax colored pencils , crayons , charcoal , chalk , pastels , various kinds of erasers , markers , styluses , various
metals ( such as silverpoint ) and electronic drawing . A drawing instrument releases a small amount of material onto a surface , leaving a visible mark . The
most common support for drawing is paper , although other materials , such as cardboard , plastic , leather , canvas , and board , may be used . Temporary
drawings may be made on a blackboard or whiteboard or indeed almost anything . The medium has been a popular and fundamental means of public expression
throughout human history . It is one of the simplest and most efficient means of communicating visual ideas . The wide availability of drawing instruments makes
drawing one of the most common artistic activities .
</item 6>

<item 7>
Drawing
Pen and ink drawings often use hatching -- groups of parallel lines . Cross-hatching uses hatching in two or more different directions to create a darker tone .
Broken hatching , or lines with intermittent breaks , form lighter tones -- and controlling the density of the breaks achieves a gradation of tone . Stippling
uses dots to produce tone , texture and shade . Different textures can be achieved depending on the method used to build tone . Drawings in dry media often use
similar techniques , though pencils and drawing sticks can achieve continuous variations in tone . Typically a drawing is filled in based on which hand the
artist favors . A right - handed artist draws from left to right to avoid smearing the image . Erasers can remove unwanted lines , lighten tones , and clean up
stray marks . In a sketch or outline drawing , lines drawn often follow the contour of the subject , creating depth by looking like shadows cast from a light in
the artist 's position . Sometimes the artist leaves a section of the image untouched while filling in the remainder .
</item 7>

<item 8>
Drawing
The wide availability of drawing instruments makes drawing one of the most common artistic activities . In addition to its more artistic forms , drawing is
frequently used in commercial illustration , animation , architecture , engineering and technical drawing . A quick , freehand drawing , usually not intended as
a finished work , is sometimes called a sketch . An artist who practices or works in technical drawing may be called a drafter , draftsman or a draughtsman .
Contents ( hide ) 1 Overview 2 History 2.1 Drawing in communication 2.2 Drawing in manuscripts 2.3 Drawing in science 2.4 Drawing as artistic expression 3
Notable artists and draftsmen 4 Materials 5 Technique 6 Tone 7 Form and proportion 8 Perspective 9 Artistry 10 Process 11 Seven Ps of drawing 12 See also 13
References 14 External links Overview ( edit ) Galileo Galilei . Phases of the Moon . 1616 . Drawing is one of the major forms of expression within the visual
arts . It is generally concerned with the marking of lines and areas of tone onto paper / other material , where the accurate representation of the visual world
is expressed upon a plane surface .
</item 8>

<item 9>
Drawing
There are also many drawing methods , such as line drawing , stippling , shading , the surrealist method of entopic graphomania ( in which dots are made at the
sites of impurities in a blank sheet of paper , and lines are then made between the dots ) , and tracing ( drawing on a translucent paper , such as tracing
paper , around the outline of preexisting shapes that show through the paper ) . A quick , unrefined drawing may be called a sketch . In fields outside art ,
technical drawings or plans of buildings , machinery , circuitry and other things are often called `` drawings '' even when they have been transferred to
another medium by printing . History ( edit ) Drawing in communication ( edit ) Drawing is one of the oldest forms of human expression , with evidence for its
existence preceding that of written communication . It is believed that drawing was used as a specialised form of communication before the invention of the
written language , demonstrated by the production of cave and rock paintings around 30,000 years ago ( Art of the Upper Paleolithic ) . These drawings , known
as pictograms , depicted objects and abstract concepts .
</item 9>

<item 10>
Drawing
Traditional drawings were monochrome , or at least had little colour , while modern colored - pencil drawings may approach or cross a boundary between drawing
and painting . In Western terminology , drawing is distinct from painting , even though similar media often are employed in both tasks . Dry media , normally
associated with drawing , such as chalk , may be used in pastel paintings . Drawing may be done with a liquid medium , applied with brushes or pens . Similar
supports likewise can serve both : painting generally involves the application of liquid paint onto prepared canvas or panels , but sometimes an underdrawing is
drawn first on that same support . Drawing is often exploratory , with considerable emphasis on observation , problem - solving and composition . Drawing is
also regularly used in preparation for a painting , further obfuscating their distinction . Drawings created for these purposes are called studies Madame
Palmyre with Her Dog , 1897 . Henri de Toulouse - Lautrec There are several categories of drawing , including figure drawing , cartooning , doodling , and free
hand .
</item 10>
=== END ITEM id=baada87941d2cda2 ===

=== ITEM id=e7e5fa4a06ac617c ===
Gold sentences and facts:
Sentence 0: The Astros played in the NL from 1962 to 2012 . They played in the West Division from 1969 to 1993 , and the Central Division from 1994 to 2012 . While a member of the NL , the Astros played in one World Series , in 2005 , against the Chicago White Sox , in which they were swept in four games . In 2017 , they became the first franchise in MLB history to have won a pennant in both the NL and the AL , when they defeated the New York Yankees in the ALCS . They subsequently won the 2017 World Series against the Los Angeles Dodgers , winning four games to three , earning the team , and Texas , its first World Series title .
  Fact 0.0: The Astros played in the NL from 1962 to 2012.
  Fact 0.1: The Astros played in the West Division from 1969 to 1993.
  Fact 0.2: The Astros played in the Central Division from 1994 to 2012.
  Fact 0.3: While a member of the NL, the Astros played in one World Series, in 2005.
  Fact 0.4: The Astros played the 2005 World Series against the Chicago White Sox.
  Fact 0.5: The Astros were swept in four games in the 2005 World Series.
  Fact 0.6: In 2017, the Astros became the first franchise in MLB history to have won a pennant in both the NL and the AL.
  Fact 0.7: In 2017, the Astros defeated the New York Yankees in the ALCS.
  Fact 0.8: The Astros subsequently won the 2017 World Series against the Los Angeles Dodgers.
  Fact 0.9: The Astros won the 2017 World Series by winning four games to three.
  Fact 0.10: The 2017 World Series title was the first World Series title for the team.
  Fact 0.11: The 2017 World Series title was the first World Series title for Texas.

Retrieved items:
<item 1>
2017 World Series
Houston Astros ( edit ) Main article : 2017 Houston Astros season The Astros wore a patch during the 2017 World Series in support of Hurricane Harvey victims in
Houston With a 101 -- 61 regular season record , the team won its first American League West title since moving from the National League Central starting in its
2013 season , and their first division title since 2001 . In the American League Division Series , they defeated the Boston Red Sox in four games and then
defeated the New York Yankees in the American League Championship Series ( ALCS ) in seven games . This was their second World Series appearance and first since
2005 , when they were swept in four games by the Chicago White Sox . They became the first team in history to make it to the World Series as members of both the
National League and the American League . The city of Houston in August 2017 suffered record flooding from Hurricane Harvey . The team began to wear patches
which had the logo of the team with the word `` Strong '' on the bottom of the patch , as well as promoting the hashtag Houston Strong . Manager A.J.
</item 1>

<item 2>
2005 World Series
1952 1953 1954 1955 1956 1957 1958 1959 1960s 1960 1961 1962 1963 1964 1965 1966 1967 1968 1969 1970s 1970 1971 1972 1973 1976 1977 1978 1979 1980s 1980 1981
1982 1984 1985 1986 1988 1989 1990s 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2010s 2011 2012 2013 2014
2015 2016 2017 2018 Houston Astros Formerly the Houston Colt .
</item 2>

<item 3>
2017 World Series
In the bottom of the inning , Yasiel Puig hit a home run off of Ken Giles and Enrique Hernández drove in Logan Forsythe to tie the game , with the latter being
the Dodgers ' first run that was not driven in by a home run . In the next inning , George Springer hit a two - run home run for the Astros off of Brandon
McCarthy to retake the lead . In the bottom of the 11th inning , Charlie Culberson homered off of Chris Devenski , who later struck out Puig to end the game .
This was the first ever World Series game in which a team hit home runs in the ninth , tenth and eleventh inning . The teams set a new record for combined home
runs in a single World Series game with eight and this was the first time in MLB history , regular season or postseason , that five home runs were hit in extra
innings . The Astros won their first World Series game in franchise history as they had been swept in their previous appearance in 2005 . Game 3 ( edit ) Lance
McCullers Jr. earned the win in Game 3 .
</item 3>

<item 4>
2005 World Series
The Astros were attempting to become the fourth consecutive wild card team to win the Series , following the Anaheim Angels ( 2002 ) , Florida Marlins ( 2003 )
and Boston Red Sox ( 2004 ) . Both teams were attempting to overcome decades of disappointment , with a combined 132 years between the two teams without a title
. The Astros were making their first Series appearance in 44 years of play , while the White Sox had waited exactly twice as long for a title , having last won
the Series in 1917 , and had not been in the Series since 1959 , three years before the Astros ' inaugural season . Like the 1982 World Series between the St.
Louis Cardinals and the Milwaukee Brewers , the 2005 World Series is one of only two World Series in the modern era ( 1903 -- present ) with no possibility for
a rematch between the two opponents , because the Astros moved to the American League in 2013 . However , the Brewers did meet the Cardinals in the 2011
National League Championship Series .
</item 4>

<item 5>
2017 World Series
Morton pitched four innings of relief to earn the win , as Corey Seager grounded out to José Altuve , who threw to Yuli Gurriel to end the game , series ,
baseball season , with the Astros winning their first championship in franchise history , and ending their 56 - year drought . The Astros became the third
consecutive road team to win a World Series Game 7 as well as the fourth consecutive team to end a World Series on the road ( after the 2014 Giants , 2015
Royals and 2016 Cubs ) . They also became just the second team to win a seven game League Championship Series and a seven game World Series in the same year ,
following the 1985 Kansas City Royals . Springer won the World Series MVP Award . Composite line score ( edit ) 2017 World Series ( 4 -- 3 ) : Houston Astros
beat Los Angeles Dodgers .
</item 5>

<item 6>
2017 World Series
With the Astros and Dodgers having waited a combined 84 years for a championship ( the Dodgers last won the World Series in 1988 , and the Astros had never won
since their 1962 formation ) , this was the third straight year in which the World Series teams had waited a combined 50 years or more to win a World Series ,
following 2015 at 59 years and 2016 at 176 years . Both teams had not lost at home this postseason leading to the World Series . The two teams did not meet in
interleague play during the regular season . Los Angeles Dodgers ( edit ) Main article : 2017 Los Angeles Dodgers season The Dodgers held a 91 -- 36 record
through August 25 . However , a late season slump , in which the team lost 20 of 25 games ( highlighted by 11 straight losses ) , led to questions about whether
the Dodgers would succeed in the postseason . Despite the slump , the team recovered to win eight of its final ten regular season games , securing their fifth
consecutive National League West title and home field advantage throughout the playoffs .
</item 6>

<item 7>
Houston Astros
They finished 101 -- 61 , with a 21 - game lead in the division , and faced the Red Sox in the first round of the AL playoffs . The Astros defeated the Red Sox
three games to one , and advanced to the American League Championship Series against the New York Yankees . The Astros won the ALCS four games to three , and
advanced to the World Series to play against the Los Angeles Dodgers . The Astros defeated the Dodgers in the deciding seventh game of the World Series ,
winning the first championship in franchise history . The city of Houston celebrated the team 's accomplishment with a parade on the afternoon of November 3 ,
2017 . Houston 's Independent School District gave the students and teachers the day off to watch the parade . On November 16 , 2017 Jose Altuve was named the
American League Most Valuable Player , capping off a historic season in which he accumulated 200 hits for the fourth consecutive season , led the majors with a
. 346 BA , and was the unquestioned clubhouse leader of the World Series Champions .
</item 7>

<item 8>
2017 World Series
Contents ( hide ) 1 Background 1.1 Los Angeles Dodgers 1.2 Houston Astros 2 Summary 3 Pre-game ceremonies 4 Game summaries 4.1 Game 1 4.2 Game 2 4.3 Game 3 4.4
Game 4 4.5 Game 5 4.6 Game 6 4.7 Game 7 4.8 Composite line score 5 Broadcasting 5.1 Television 5.1. 1 Ratings 5.2 Radio 6 Sponsorship 7 See also 8 Notes 9
References 10 External links Background ( edit ) This was the first World Series matchup , and second postseason meeting overall , between the Astros and
Dodgers . Los Angeles defeated Houston in the 1981 National League Division Series in five games en route to its World Series championship that year . The teams
also met in the 1980 National League West tie - breaker game . This was the first Fall Classic since 1970 , and the eighth overall , in which both participants
had 100 or more wins during the regular season .
</item 8>

<item 9>
Graph query row:
organization: Houston Astros
relationship: PARTICIPATED_IN
event: 2017 World Series
date: 2017
result: won four games to three
outcome: won
</item 9>

<item 10>
Houston Astros
In 2017 , they became the first franchise in MLB history to have won a pennant in both the NL and the AL , when they defeated the New York Yankees in the ALCS .
They subsequently won the 2017 World Series against the Los Angeles Dodgers , winning four games to three , earning the team , and Texas , its first World
Series title . Contents ( hide ) 1 Franchise history 1.1 Major League Baseball comes to Texas 1.2 1962 -- 1964 : The Colt . 45s 1.3 1965 -- 1970 : The Great
Indoors 1.4 1971 -- 1974 : The boys in orange 1.4.
</item 10>
=== END ITEM id=e7e5fa4a06ac617c ===

=== ITEM id=c996590513a433e8 ===
Gold sentences and facts:
Sentence 0: Cast Character Rank / Position Seasons Notes Regular Recurring Guest Christopher Meloni Elliot Stabler Detective 1 -- 12 Mariska Hargitay Olivia Benson Lieutenant 1 -- Richard Belzer John Munch DA Investigator 1 -- 15 15 , 17 Dann Florek Donald Cragen Captain 1 -- 15 16 Michelle Hurd Monique Jeffries Detective 1 -- 2 Stephanie March Alexandra Cabot Assistant District Attorney 2 -- 5 , 11 2 , 10 , 13 6 Ice - T Odafin `` Fin '' Tutuola Detective 2 -- Tamara Tunie Melinda Warner Medical Examiner 7 -- 12 2 -- 6 , 13 -- 17 19 B.D. Wong George Huang FBI Special Agent 4 -- 12 2 -- 3 13 -- 15 , 17 Diane Neal Casey Novak Senior Assistant District Attorney 5 -- 9 13 12 Adam Beach Chester Lake Detective 9 8 Michaela McManus Kim Greylek Assistant District Attorney 10 Danny Pino Nick Amaro Detective 13 -- 16 Kelli Giddish Amanda Rollins Detective 13 -- Raúl Esparza Rafael Barba Assistant District Attorney 15 -- 14 Peter Scanavino Dominick Carisi Jr . Detective 16 -- 16
  Fact 0.0: Christopher Meloni plays Elliot Stabler.
  Fact 0.1: Elliot Stabler is a Detective.
  Fact 0.2: Christopher Meloni's season entry is 1 to 12.
  Fact 0.3: Mariska Hargitay plays Olivia Benson.
  Fact 0.4: Olivia Benson is a Lieutenant.
  Fact 0.5: Mariska Hargitay's season entry starts at season 1.
  Fact 0.6: Richard Belzer plays John Munch.
  Fact 0.7: John Munch is a DA Investigator.
  Fact 0.8: Richard Belzer's season entries are 1 to 15 and 15 and 17.
  Fact 0.9: Dann Florek plays Donald Cragen.
  Fact 0.10: Donald Cragen is a Captain.
  Fact 0.11: Dann Florek's season entries are 1 to 15 and 16.
  Fact 0.12: Michelle Hurd plays Monique Jeffries.
  Fact 0.13: Monique Jeffries is a Detective.
  Fact 0.14: Michelle Hurd's season entry is 1 to 2.
  Fact 0.15: Stephanie March plays Alexandra Cabot.
  Fact 0.16: Alexandra Cabot is an Assistant District Attorney.
  Fact 0.17: Stephanie March's season entries are 2 to 5, 11, 2, 10, 13, and 6.
  Fact 0.18: Ice - T plays Odafin "Fin" Tutuola.
  Fact 0.19: Odafin "Fin" Tutuola is a Detective.
  Fact 0.20: Ice - T's season entry starts at season 2.
  Fact 0.21: Tamara Tunie plays Melinda Warner.
  Fact 0.22: Melinda Warner is a Medical Examiner.
  Fact 0.23: Tamara Tunie's season entries are 7 to 12, 2 to 6, 13 to 17, and 19.
  Fact 0.24: B.D. Wong plays George Huang.
  Fact 0.25: George Huang is an FBI Special Agent.
  Fact 0.26: B.D. Wong's season entries are 4 to 12, 2 to 3, 13 to 15, and 17.
  Fact 0.27: Diane Neal plays Casey Novak.
  Fact 0.28: Casey Novak is a Senior Assistant District Attorney.
  Fact 0.29: Diane Neal's season entries are 5 to 9, 13, and 12.
  Fact 0.30: Adam Beach plays Chester Lake.
  Fact 0.31: Chester Lake is a Detective.
  Fact 0.32: Adam Beach's season entries are 9 and 8.
  Fact 0.33: Michaela McManus plays Kim Greylek.
  Fact 0.34: Kim Greylek is an Assistant District Attorney.
  Fact 0.35: Michaela McManus's season entry is 10.
  Fact 0.36: Danny Pino plays Nick Amaro.
  Fact 0.37: Nick Amaro is a Detective.
  Fact 0.38: Danny Pino's season entry is 13 to 16.
  Fact 0.39: Kelli Giddish plays Amanda Rollins.
  Fact 0.40: Amanda Rollins is a Detective.
  Fact 0.41: Kelli Giddish's season entry starts at season 13.
  Fact 0.42: Raúl Esparza plays Rafael Barba.
  Fact 0.43: Rafael Barba is an Assistant District Attorney.
  Fact 0.44: Raúl Esparza's season entry is 15 to 14.
  Fact 0.45: Peter Scanavino plays Dominick Carisi Jr.
  Fact 0.46: Dominick Carisi Jr. is a Detective.
  Fact 0.47: Peter Scanavino's season entry is 16 to 16.

Retrieved items:
<item 1>
Law & Order: Special Victims Unit
On May 12 , 2017 , the series was renewed for a nineteenth season , which premiered on September 27 , 2017 . As of January 17 , 2018 , Law & Order : Special
Victims Unit has aired 421 original episodes . Contents ( hide ) 1 Production 1.1 History and development 1.1. 1 Executive producers 1.2 Filming 1.3 Broadcast
history 1.4 Russian adaptation 2 Casting 2.1 Salaries 3 Cast and characters 4 Series overview 5 Episodes 6 Reception 6.1 U.S. television ratings 6.2 Awards and
honors 7 Broadcast and streaming 7.1 Syndication 8 International broadcast 9 References 9.1 Notes 9.2 Bibliography 10 External links Production ( edit ) History
and development ( edit ) The idea for Law & Order : Special . Victims Unit originated with the 1986 `` preppie murder '' case of Robert Chambers , who strangled
Jennifer Levin , a woman he dated whom he later killed during what he claimed was consensual `` rough sex '' in Manhattan 's Central Park .
</item 1>

<item 2>
Law & Order: Special Victims Unit
Wong , Stephanie March Cast of season 9 ( 2007 -- 08 ) , from left : Diane Neal , Wong , Tamara Tunie , Meloni , Florek , Hargitay , Ice - T , Belzer , Adam
Beach Cast of season 10 ( 2008 -- 09 ) from left : Michaela McManus , Wong , Tunie , Meloni , Hargitay , Florek , Belzer , Ice - T The cast of season 15 ( 2013
-- 2014 ) from left : Raúl Esparza , Florek , Danny Pino , Hargitay , Kelli Giddish , Belzer , Ice - T The cast of seasons 17 -- 18 ( 2015 -- 2017 ) . from left
: Esparza , Giddish , Hargitay , Ice - T , Peter Scanavino Casting for the lead characters of Law & Order : Special Victims Unit occurred in the spring of 1999
. Dick Wolf , along with officials from NBC and Studios USA were at the final auditions for the two leads at Rockefeller Center .
</item 2>

<item 3>
Law & Order: Special Victims Unit
1999 ( 1999 - 09 - 20 ) -- present Chronology Related shows Law & Order franchise External links Website Law & Order : Special Victims Unit ( often abbreviated
to Law & Order : SVU or just SVU ) is an American police procedural , legal , crime drama television series set in New York City , where it is also primarily
produced . In the style of the original Law & Order , episodes are often `` ripped from the headlines '' or loosely based on real crimes that have received
media attention . Created by Dick Wolf , the series premiered on NBC on September 20 , 1999 , as the second series in Wolf 's successful Law & Order franchise .
Law & Order : Special Victims Unit has been nominated for and won numerous awards , including the 2006 Emmy for Outstanding Lead Actress in a Drama Series for
Mariska Hargitay , the first , and to date only , Emmy to be received by a regular on any Law & Order series . It is the current longest running scripted
non-animated U.S.
</item 3>

<item 4>
Law & Order: Special Victims Unit
Series overview ( edit ) In the criminal justice system , sexually based offenses are considered especially heinous . In New York City , the dedicated
detectives who investigate these vicious felonies are members of an elite squad known as the Special Victims Unit . These are their stories . -- Opening
narration spoken by Steven Zirnkilton Based out of the New York City Police Department 's 16th precinct in Manhattan , Law & Order : Special Victims Unit delves
into the dark side of the New York underworld as the detectives of a new elite force , the Special Victims Unit ( SVU for short ) , investigate and prosecute
various sexually - oriented crimes including rape , pedophilia , and domestic violence . They also investigate the abuses of children , the disabled and elderly
victims of non-sexual crimes who require specialist handling . All the while trying to balance the effects of the investigation on their own lives . Its stories
also touch on the political and societal issues associated with gender identity , sexual preferences , and equality rights . While the victim is often murdered
, this is not always the case , and victims frequently play prominent roles in episodes .
</item 4>

<item 5>
Law & Order: Special Victims Unit
Law & Order : Special Victims Unit - Wikipedia Law & Order : Special Victims Unit Jump to : navigation , search For the current season , see Law & Order :
Special Victims Unit ( season 19 ) . Law & Order : Special Victims Unit Also known as Law & Order : SVU SVU Genre Police procedural Legal drama Thriller Created
by Dick Wolf Starring Christopher Meloni Mariska Hargitay Richard Belzer Dann Florek Michelle Hurd Stephanie March Ice - T B.D. Wong Diane Neal Tamara Tunie
Adam Beach Michaela McManus Danny Pino Kelli Giddish Raúl Esparza Peter Scanavino Opening theme Theme of Law & Order : Special Victims Unit Composer ( s ) Mike
Post Country of origin United States Original language ( s ) English No. of seasons 19 No. of episodes 421 ( list of episodes ) Production Executive producer (
s ) Dick Wolf Robert Palm ( season 1 ) David J.
</item 5>

<item 6>
Law & Order: Special Victims Unit
It is the current longest running scripted non-animated U.S. primetime TV series since the cancellation of the original Law & Order on May 24 , 2010 and is the
fourth - longest running scripted U.S. primetime TV series on a major broadcast network . Law & Order : Special Victims Unit originally centered almost
exclusively on the detectives of the Special Victims Unit in a fictional version of the 16th Precinct of the New York City Police Department . As the series
progressed , additional supporting characters were added as allies of the detectives in the New York County Manhattan District Attorney 's office ( known as
advisers from the Sex Crimes Bureau ) and the Medical Examiner 's office . Certain episodes will go into detail about detectives ' personal lives and how they
may or may not tie into the crimes dealt with during the show . Typical episodes follow the detectives and their colleagues as they investigate and prosecute
sexually based offenses , child abuse , and crimes against the elderly . The show starred Christopher Meloni as Detective Elliot Stabler and Mariska Hargitay as
Detective Olivia Benson for its first twelve seasons until the former left the cast , unable to come to an agreement on his contract .
</item 6>

<item 7>
Law & Order: Special Victims Unit
Cast and characters ( edit ) Main article : List of Law & Order : Special Victims Unit characters Cast Character Rank / Position Seasons Notes Regular Recurring
Guest Christopher Meloni Elliot Stabler Detective 1 -- 12 Mariska Hargitay Olivia Benson Lieutenant 1 -- Richard Belzer John Munch DA Investigator 1 -- 15 15 ,
17 Dann Florek Donald Cragen Captain 1 -- 15 16 Michelle Hurd Monique Jeffries Detective 1 -- 2 Stephanie March Alexandra Cabot Assistant District Attorney 2 --
5 , 11 2 , 10 , 13 6 Ice - T Odafin `` Fin '' Tutuola Detective 2 -- Tamara Tunie Melinda Warner Medical Examiner 7 -- 12 2 -- 6 , 13 -- 17 19 B.D.
</item 7>

<item 8>
Law & Order: Special Victims Unit
The unit also works with the Manhattan District Attorney 's office as they prosecute cases and seek justice for SVU 's victims and survivors with precision and
a passion to win and bring closure to the intense investigations . The series often uses stories that are `` ripped from the headlines '' or based on real
crimes . Such episodes take a real crime and fictionalize it by changing some details . Originally the show focused around the detective pairings of Elliot
Stabler & Olivia Benson and John Munch & Brian Cassidy . Stabler is a seasoned veteran of the unit who has seen it all and tries his best to protect his family
from the horrors he sees every day . His partner Benson 's difficult past as the child of a rape victim is the reason she joined the unit . Backing them up is
John Munch , and his first partner Brian Cassidy . Munch is a transfer from Baltimore 's homicide unit , who brings his acerbic wit , conspiracy theories , and
street - honed investigative skills . Whilst Cassidy is young and eager to learn from his fellow detectives . These two detective teams received support from
Detectives Monique Jeffries and Ken Briscoe .
</item 8>

<item 9>
Law & Order: Special Victims Unit
52 8.31 18 21 September 21 , 2016 ( 2016 - 09 - 21 ) 7.83 May 24 , 2017 ( 2017 - 05 - 24 ) 6.22 2016 -- 17 No. 48 7.39 19 24 September 27 , 2017 ( 2017 - 09 -
27 ) 5.67 TBA TBA 2017 - 18 TBA TBA Awards and honors ( edit ) Main article : List of awards and nominations received by Law & Order : Special Victims Unit Law
& Order : Special Victims Unit has received many awards and award nominations . Mariska Hargitay has twice been nominated for a Golden Globe Award and won once
in 2005 . The show has been nominated numerous times for the Emmy Award . Mariska Hargitay has been nominated for the Outstanding Lead Actress in a Drama Series
category 8 years in a row beginning in 2004 and won the Emmy in 2006 . Christopher Meloni was nominated for the Outstanding Lead Actor in a Drama Series
category in 2006 .
</item 9>

<item 10>
Law & Order: Special Victims Unit
External links ( edit ) Wikimedia Commons has media related to Law & Order : Special Victims Unit . Wikiquote has quotations related to : Law & Order : Special
Victims Unit Official website on NBC Law & Order : Special Victims Unit on IMDb Law & Order : Special Victims Unit at TV.com `` List of Law & Order : SVU
Episodes '' . TV Guide . Law & Order : Special Victims Unit on MyNetworkTV Law & Order franchise Series Law & Order ( 1990 -- 2010 ) Law & Order : Special
Victims Unit ( 1999 -- present ) Law & Order : Criminal Intent ( 2001 -- 2011 ) Law & Order : Trial by Jury ( 2005 -- 2006 ) Law & Order : LA ( 2010 -- 2011 )
Law & Order True Crime ( 2017 -- present ) Adaptations Paris enquêtes criminelles ( 2007 -- 2008 ) Law & Order : UK ( 2009 -- 2014 ) Video games Law & Order :
Dead on the Money Law & Order : Double or Nothing Law & Order : Justice Is Served Law & Order : Criminal Intent Law & Order : Legacies Related series Homicide :
</item 10>
=== END ITEM id=c996590513a433e8 ===

=== ITEM id=7198e6607d695c79 ===
Gold sentences and facts:
Sentence 0: Name Age Hometown Boarded Destination Lifeboat Body Allen , Miss Elizabeth Walton 29 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Allison , Mr. Hudson Joshua Creighton 30 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 135 and chauffeur , Mr. George Swane , 19 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 294 and cook , Miss Amelia Mary `` Mildred '' Brown 18 London , England , UK Southampton Montreal , Quebec , Canada 11 Allison , Mrs. Bessie Waldo ( née Daniels ) 25 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada and maid , Miss Sarah Daniels 33 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 8 Allison , Miss Helen Loraine Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Allison , Master Hudson Trevor 0.11 ! 11 mo . Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 11 and nurse , Miss Alice Catherine Cleaver 22 London , England , UK Southampton Montreal , Quebec , Canada 11 Anderson , Mr. Harry 47 New York City , New York , US Southampton New York City , New York , US Andrews , Miss Kornelia Theodosia 62 Hudson , New York , US Cherbourg Hudson , New York , US 10 Andrews , Mr. Thomas 39 Belfast , Ireland , UK Belfast New York City Appleton , Mrs. Charlotte Lane ( née Lamson ) 53 New York City Southampton New York City Artagaveytia , Mr. Ramon 71 Buenos Aires , Argentina Cherbourg New York City 22 Astor , Colonel John Jacob IV 47 New York City Cherbourg New York City 124 Astor , Mrs. Madeleine Talmage ( née Force ) 18 New York City Cherbourg New York City and maid , Miss Rosalie Bidois 46 New York City Cherbourg New York City and nurse , Miss Caroline Louise Endres 39 New York City Cherbourg New York City Aubart , Mrs. Léontine Pauline 24 Paris , France Cherbourg New York City 9 and maid , Miss Emma Sägesser 24 Paris , France Cherbourg New York City 9 Barkworth , Mr. Algernon Henry Wilson 47 Hessle , East Yorkshire , England , UK Southampton New York City Baumann , Mr. John D . 60 New York City Cherbourg New York City Baxter , Mrs. Hélène ( née de Lanaudière - Chaput ) 50 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada 6 Baxter , Mr. Quigg Edmond 24 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada Beattie , Mr. Thomson 36 Fergus , Ontario , Canada Southampton Fergus , Ontario , Canada 331 Beckwith , Mr. Richard Leonard 37 New York City Southampton New York City 5 Beckwith , Mrs. Sallie ( née Monypeny ) 46 New York City Southampton New York City 5 Behr , Mr. Karl Howell 26 New York City Cherbourg New York City 5 Birnbaum , Mr. Jakob 24 Antwerp , Belgium Cherbourg San Francisco , California , US 148 Bishop , Mr. Dickinson H. `` Dick '' 25 Dowagiac , Michigan , US Cherbourg Dowagiac , Michigan , US 7 Bishop , Mrs. Helen ( née Walton ) 19 Dowagiac , Michigan , US Cherbourg Dowagiac , Michigan , US 7 Björnström - Steffanson , Mr. Mauritz Håkan 28 Stockholm , Sweden Southampton Washington , D.C. , US Blackwell , Mr. Stephen Weart 45 Trenton , New Jersey , US Southampton Trenton , New Jersey , US Blank , Mr. Henry 39 Glen Ridge , New Jersey , US Cherbourg Glen Ridge , New Jersey , US 7 Bonnell , Miss Elizabeth 61 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Bonnell , Miss Caroline 30 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Borebank , Mr. John James 42 London , England , UK Southampton Toronto , Ontario , Canada Bowerman , Miss Elsie Edith 22 St. Leonards - on - Sea , East Sussex , England , UK Southampton New York City 6 Brady , Mr. John Bertram 41 Pomeroy , Washington , US Southampton Pomeroy , Washington , US Brandeis , Mr. Emil 48 Omaha , Nebraska , US Cherbourg Omaha , Nebraska , US 208 Brereton , Mr. George Andrew ( alias George A. Brayton ) 37 Los Angeles , US Southampton Los Angeles , California , US 9 Brewe , Dr. Arthur Jackson 45 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US Brown , Mrs. Caroline Lane ( née Lamson ) 59 Belmont , Massachusetts , US Southampton Belmont , Massachusetts , US Brown , Mrs. Margaret ( née Tobin ) 44 Denver , Colorado , US Cherbourg Denver , Colorado , US 6 Bucknell , Mrs. Emma Eliza ( née Ward ) 59 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 8 and maid , Miss Albina Bazzani 36 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 8 Butt , Major Archibald Willingham 46 Washington , D.C. , US Southampton Washington , D.C. , US Calderhead , Mr. Edward Pennington 42 New York City Southampton New York City 5 Candee , Mrs. Helen Churchill ( née Hungerford ) 53 Washington , D.C. , US Cherbourg Washington , D.C. , US 6 Cardeza , Mrs. Charlotte Wardle ( née Drake ) 58 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US and maid , Miss Annie Moore Ward 38 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US Cardeza , Mr. Thomas Drake Martinez 36 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US and valet , Mr. Gustave J. Lesueur 35 Germantown , Pennsylvania , US Cherbourg Germantown , Pennsylvania , US Carlsson , Mr. Frans Olof 33 New York City Southampton New York City Carrau , Mr. Francisco M . 31 Montevideo , Uruguay Southampton Montevideo , Uruguay Carrau , Mr. José Pedro 17 Montevideo , Uruguay Southampton Montevideo , Uruguay Carter , Mr. William Ernest 36 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and valet , Mr. Alexander Cairns 28 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and chauffeur , Mr. Charles Augustus Aldworth 30 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Mrs. Lucile ( née Polk ) 36 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US and maid , Miss Auguste Serreplan 30 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Miss Lucile Polk 13 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Carter , Master William Thornton II 11 Bryn Mawr , Pennsylvania , US Southampton Bryn Mawr , Pennsylvania , US Case , Mr. Howard Brown 49 Ascot , Berkshire , England , UK Southampton Rochester , New York , US Cassebeer , Mrs. Eleanor Genevieve ( née Fosdick ) 36 New York City Cherbourg New York City 5 Cavendish , Mr. Tyrell William 36 London , England , UK Southampton New York City 172 Cavendish , Mrs. Julia Florence ( née Siegel ) 25 London , England , UK Southampton New York City 6 and maid , Miss Ellen `` Nellie '' Barber 26 London , England , UK Southampton New York City 6 Chaffee , Mr. Hubert Fuller 46 Amenia , North Dakota , US Southampton Amenia , North Dakota , US Chaffee , Mrs. Carrie Constance ( née Toogood ) 47 Amenia , North Dakota , US Southampton Amenia , North Dakota , US Chambers , Mr. Norman Campbell 27 New York City Southampton New York City 5 Chambers , Mrs. Bertha ( née Griggs ) 32 New York City Southampton New York City 5 Cherry , Miss Gladys 30 London , England , UK Southampton Vancouver , British Columbia , Canada 8 Chevré , Mr. Paul Romaine Marie Léonce 45 Paris , France Cherbourg Ottawa , Ontario , Canada 7 Chibnall , Mrs. Edith Martha Bowerman ( née Barber ) 48 St. Leonards - on - Sea , East Sussex , England , UK Southampton New York City 6 Chisholm , Mr. Roderick Robert Crispin 40 Belfast , Ireland , UK Belfast New York City Clark , Mr. Walter Miller 27 Los Angeles , California , US Cherbourg Los Angeles , California , US Clark , Mrs. Virginia Estelle ( née McDowell ) 26 Los Angeles , California , US Cherbourg Los Angeles , California , US Clifford , Mr. George Quincy 40 Stoughton , Massachusetts , US Southampton Stoughton , Massachusetts Colley , Mr. Edward Pomeroy 37 Dublin , Ireland , UK Southampton Vancouver , British Columbia , Canada Compton , Mrs. Mary Eliza ( née Ingersoll ) 64 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US 14 Compton , Miss Sara Rebecca 39 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US 14 Compton , Mr. Alexander Taylor Jr . 37 Lakewood , New Jersey , US Cherbourg Lakewood , New Jersey , US Cornell , Mrs. Malvina Helen ( née Lamson ) 55 New York City Southampton New York City Crafton , Mr. John Bertram 59 Roachdale , Indiana , US Southampton Roachdale , Indiana , US Crosby , Captain Edward 70 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 269 Crosby , Mrs. Catherine Elizabeth ( née Halstead ) 64 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 7 Crosby , Miss Harriette Rebecca 39 Milwaukee , Wisconsin , US Southampton Milwaukee , Wisconsin , US 7 Cumings , Mr. John Bradley 39 New York City Cherbourg New York City Cumings , Mrs. Florence Briggs ( née Thayer ) 35 New York City Cherbourg New York , US Daly , Mr. Peter Dennis 51 Lima , Peru Southampton Lima , Peru Daniel , Mr. Robert Williams 27 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Davidson , Mr. Thornton 31 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Davidson , Mrs. Orian ( née Hays ) 27 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Dick , Mr. Albert Adrian 31 Calgary , Alberta , Canada Southampton Calgary , Alberta , Canada Dick , Mrs. Vera ( née Gillespie ) 17 Calgary , Alberta , Canada Southampton Calgary , Alberta , Canada Dineston , Mrs. Nancy May Bowles 28 London , England , UK Southampton New York City , USA Dineston , George Samuel 30 London , England , UK Southampton New York City , USA Dodge , Dr. Washington 52 San Francisco , California , US Southampton San Francisco , California , US 13 Dodge , Mrs. Ruth ( née Vidaver ) 34 San Francisco , California , US Southampton San Francisco , California , US 5 Dodge , Master Washington , Jr . San Francisco , California , US Southampton San Francisco , California , US 5 Douglas , Mr. Walter Donald 50 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US 62 Douglas , Mrs. Mahala ( née Dutton ) 48 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US and maid , Miss Berthe Leroy 27 Minneapolis , Minnesota , US Cherbourg Minneapolis , Minnesota , US Douglas , Mrs. Mary Hélène ( née Baxter ) 27 Montreal , Quebec , Canada Cherbourg Montreal , Quebec , Canada 6 Duff Gordon , Sir Cosmo Edmund 49 London , England , UK Cherbourg New York City Duff Gordon , Lucy Christiana , Lady ( née Sutherland ) 48 London , England , UK Cherbourg New York City and secretary , Miss Laura Mabel Francatelli 31 London , England , UK Cherbourg New York City Dulles , Mr. William Crothers 39 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 133 Eustis , Miss Elizabeth Mussey 54 Brookline , Massachusetts , US Cherbourg Brookline , Massachusetts , US Evans , Miss Edith Corse 36 New York City Cherbourg New York City Flegenheim , Mrs. Antoinette ( née Wendt ) 48 New York City Cherbourg New York City 7 Flynn , Mr. John Irwin 36 Brooklyn , New York , US Southampton Brooklyn , New York , US 5 Foreman , Mr. Benjamin Laventall 30 New York City Southampton New York City Fortune , Mr. Mark 64 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Fortune , Mrs. Mary ( née McDougald ) 60 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Ethel Flora 28 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Alice Elizabeth 24 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Miss Mabel Helen 23 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 10 Fortune , Mr. Charles Alexander 19 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Franklin , Mr. Thomas Parnham 37 London , England , UK Southampton New York City Frauenthal , Dr. Henry William 49 London , England , UK Southampton New York City 5 Frauenthal , Mrs. Clara ( née Heinsheimer ) 42 London , England , UK Southampton New York City 5 Frauenthal , Mr. Isaac Gerald 43 London , England , UK Southampton New York City 5 Frölicher , Mr. Maximilian Josef 60 Zürich , Switzerland Cherbourg New York City 5 Frölicher , Mrs. Margaretha Emerentia ( née Stehli ) 48 Zürich , Switzerland Cherbourg New York City 5 Frölicher - Stehli , Miss Hedwig Margaritha 22 Zürich , Switzerland Cherbourg New York City 5 Futrelle , Mr. Jacques Heath 37 Scituate , Massachusetts , US Southampton Scituate , Massachusetts , US Futrelle , Mrs. Lily May ( née Peel ) 35 Scituate , Massachusetts , US Southampton Scituate , Massachusetts , US Gee , Mr. Arthur H . 47 St. Annes - on - Sea , Lancashire , England , UK Southampton Mexico City , Mexico 275 Gibson , Mrs. Pauline Caroline ( née Boeson ) 44 New York City Cherbourg New York City 7 Gibson , Miss Dorothy Winifred 22 New York City Cherbourg New York City 7 Goldenberg , Mr. Samuel L . 47 Paris , France Cherbourg New York City 5 Goldenberg , Mrs. Nella ( née Wiggins ) 40 Paris , France Cherbourg New York City 5 Goldschmidt , Mr. George B . 71 New York City Cherbourg New York City Gracie IV , Colonel Archibald 53 Washington , D.C. , US Southampton Washington , D.C. , US Graham , Mr. George Edward 38 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada 147 Graham , Mrs. Edith Ware ( née Junkins ) 59 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US Graham , Miss Margaret Edith 19 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US and governess , Miss Elizabeth Weed Shutes 40 Greenwich , Connecticut , US Southampton Greenwich , Connecticut , US Greenfield , Mrs. Blanche ( née Strouse ) 45 New York City Cherbourg New York City 7 Greenfield , Mr. William Bertram 23 New York City Cherbourg New York City 7 Guggenheim , Mr. Benjamin 46 Paris , France Cherbourg New York City and valet , Mr. Victor Giglio 24 Paris , France Cherbourg New York City and chauffeur , Mr. René Pernot 39 Paris , France Cherbourg New York City Harder , Mr. George Achilles 25 New York City Cherbourg New York City 5 Harder , Mrs. Dorothy ( née Annan ) 21 New York City Cherbourg New York City 5 Harper , Mr. Henry Sleeper 48 New York City Cherbourg New York City and dragoman , Mr. Hammad Hassab 27 Cairo , Egypt Cherbourg New York City Harper , Mrs. Myna ( née Haxtun ) 49 New York City Cherbourg New York City Harris , Mr. Henry Birkhardt 45 New York City Southampton New York City Harris , Mrs. Irene ( née Wallach ) 35 New York City Southampton New York City Hawksford , Mr. Walter James 45 Kingston , Surrey , England , UK Southampton New York City Hays , Mr. Charles Melville 55 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 307 and clerk , Mr. Vivian Ponsonby Payne 22 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Hays , Mrs. Clara Jennings ( née Grigg ) 52 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada and maid , Miss Mary Anne Perreault 33 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Hays , Miss Margaret Bechstein 24 New York City Cherbourg New York City 7 Head , Mr. Christopher 42 Chelsea , London , England , UK Southampton New York City Hilliard , Mr. Herbert Henry 44 Brighton , Massachusetts , US Southampton Brighton , Massachusetts , US Hipkins , Mr. William Edward 55 Birmingham , West Midlands , England , UK Southampton New York City Hippach , Mrs. Ida Sophia ( née Fischer ) 44 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Hippach , Miss Jean Gertrude 17 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Hogeboom , Mrs. Anna Louisa ( née Andrews ) 51 Hudson , New York , US Cherbourg Hudson , New York , US 10 Holverson , Mr. Alexander Oskar 42 New York City Southampton New York City 38 Holverson , Mrs. Mary Aline ( née Towner ) 35 New York City Southampton New York City 8 Homer , Mr. Harry ( alias E. Haven ) 40 Indianapolis , Indiana , US Southampton Indianapolis , Indiana , US 15 Hoyt , Mr. Frederick Maxfield 35 New York City Southampton Stamford , Connecticut , US Hoyt , Mrs. Jane Anne ( née Forby ) 31 New York City Southampton Stamford , Connecticut , US Hoyt , Mr. William Fisher 42 New York City Cherbourg New York City 14 Isham , Miss Ann Elizabeth 50 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Ismay , Mr. Joseph Bruce 49 Liverpool , Merseyside , England , UK Southampton New York City and valet , Mr. John Richard Fry 39 Liverpool , Merseyside , England , UK Southampton New York City and secretary , Mr. William Henry Harrison 45 Wallasey , Merseyside , England , UK Southampton New York City 110 Johnson , Mrs. Madeline Eliza ( née Austin ) 40 London , England , UK Southampton New York , New York , US 6 Johnson ( later changed to Langston ) , Mr. Samuel James 10 London , England , UK Southampton New York , New York , US Jones , Mr. Charles Cresson 46 Bennington , Vermont , US Southampton Bennington , Vermont , US 80 Julian , Mr. Henry Forbes 50 Torquay , Devon , England , UK Southampton San Francisco , US Kent , Mr. Edward Austin 58 Buffalo , New York , US Cherbourg Buffalo , New York , US 258 Kenyon , Mr. Frederick R . 41 Pittsburgh , Pennsylvania , US Southampton Pittsburgh , Pennsylvania , US Kenyon , Mrs. Marion ( née Stauffer ) 31 Pittsburgh , Pennsylvania , US Southampton Pittsburgh , Pennsylvania , US 8 Kimball , Mr. Edwin Nelson Jr . 42 Boston , Massachusetts , US Southampton Boston , Massachusetts , US 5 Kimball , Mrs. Gertrude ( née Parsons ) 45 Boston , Massachusetts , US Southampton Boston , Massachusetts , US 5 Klaber , Mr. Herman 45 Portland , Oregon , US Southampton Portland , Oregon , US Langston , Ms. Maxine Jane 25 Duncansville , Pennsylvania , US Southampton Duncansivlle , Pennsylvania , US Lambert - Williams , Mr. Fletcher Fellows 43 London , England , UK Southampton Newark , New Jersey , US Leader , Dr. Alice ( née Farnham ) 49 New York City Southampton New York City 8 Leslie , Lucy Noël Martha , Countess of Rothes ( née Dyer - Edwardes ) 27 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 8 and maid , Miss Roberta Elizabeth Mary `` Cissy '' Maioni 20 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 8 Lewy , Mr. Ervin G . 31 Chicago , Illinois , US Cherbourg Chicago , Illinois , US Lindeberg - Lind , Mr. Erik Gustav ( alias Edward Lingrey ) 42 Jordanstorp , Södermanland , Sweden Southampton New York City Lindström , Mrs. Sigrid ( née Posse ) 55 Stockholm , Sweden Cherbourg New York City 6 Lines , Mrs. Elizabeth Lindsey ( née James ) 50 Paris , France Cherbourg Hanover , New Hampshire , US 9 Lines , Miss Mary Conover 16 Paris , France Cherbourg Hanover , New Hampshire , US 9 Long , Mr. Milton Clyde 29 Springfield , Massachusetts , US Southampton Springfield , Massachusetts , US 126 Longley , Miss Gretchen Fiske 21 Hudson , New York , US Cherbourg Hudson , New York , US 10 Loring , Mr. Joseph Holland 30 London , England , UK Southampton New York City Madill , Miss Georgette Alexandra 16 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Maguire , Mr. John Edward 30 Brockton , Massachusetts , US Southampton Brockton , Massachusetts , US Maréchal , Mr. Pierre , Sr . 28 Paris , France Cherbourg New York City 7 Marvin , Mr. Daniel Warner 18 New York City Southampton New York City Marvin , Mrs. Mary Graham Carmichael ( née Farquarson ) 18 New York City Southampton New York City 10 Mayné , Miss Bertha Antonine 24 Brussels , Belgium Cherbourg Montreal , Quebec , Canada 6 McCaffry , Mr. Thomas Francis 46 Vancouver , British Columbia , Canada Southampton Vancouver , British Columbia , Canada 292 McCarthy , Mr. Timothy J . 54 Dorchester , Massachusetts , US Southampton Dorchester , Massachusetts , US 175 McGough , Mr. James Robert 35 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US 7 Meyer , Mr. Edgar Joseph 28 New York City Cherbourg New York City Meyer , Mrs. Leila ( née Saks ) 25 New York City Cherbourg New York City 6 Millet , Mr. Francis Davis 65 East Bridgewater , Massachusetts , US Cherbourg East Bridgewater , Massachusetts , US 249 Minahan , Dr. William Edward 44 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 230 Minahan , Mrs. Lillian E. ( née Thorpe ) 37 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 14 Minahan , Miss Daisy E . 33 Fond du Lac , Wisconsin , US Southampton Fond du Lac , Wisconsin , US 14 Mock , Mr. Philipp Edmund 30 New York City Cherbourg New York City 11 Molson , Mr. Harry Markland 55 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada Moore , Mr. Clarence Bloomfield 47 Washington , D.C. , US Southampton Washington , D.C. , US and valet , Mr. Charles Henry Harrington 37 Washington , D.C. , US Southampton Washington , D.C. , US Natsch , Mr. Charles 36 Brooklyn , New York , US Cherbourg Brooklyn , New York , US Newell , Mr. Arthur Webster 58 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 122 Newell , Miss Madeleine 31 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 6 Newell , Miss Marjorie Anne 23 Lexington , Massachusetts , US Cherbourg Lexington , Massachusetts , US 6 Newsom , Miss Helen Monypeny 19 New York City Southampton New York City 5 Nicholson , Mr. Arthur Ernest 64 Shanklin , Isle of Wight , England , UK Southampton New York City 263 Nourney , Mr. Alfred 20 Cologne , German Empire Cherbourg New York City 7 Omont , Mr. Alfred Fernand 29 Le Havre , France Cherbourg New York City 7 Ostby , Mr. Engelhart Cornelius 64 Providence , Rhode Island , US Southampton Providence , Rhode Island , US 234 Ostby , Miss Helene Ragnhild 22 Providence , Rhode Island , US Southampton Providence , Rhode Island , US 5 Ovies y Rodriguez , Mr. Servando José Florentino 36 Havana , Cuba Cherbourg Havana , Cuba 189 Parr , Mr. William Henry Marsh 29 Belfast , Ireland , UK Belfast New York City Partner , Mr. Austin 40 Tolworth , London , England , UK Southampton Toronto , Ontario , Canada 166 Pears , Mr. Thomas Clinton 29 Isleworth , London , England , UK Southampton New York City Pears , Mrs. Edith ( née Wearne ) 22 Isleworth , London , England , UK Southampton New York City 8 Peñasco y Castellana , Mr. Victor 24 Madrid , Spain Cherbourg New York City Peñasco y Castellana , Mrs. Maria Josefa ( née Perez de Soto y Vallejo ) 22 Madrid , Spain Cherbourg New York City 8 and maid , Doña Fermina Oliva y Ocana 39 Madrid , Spain Cherbourg New York City 8 Peuchen , Major Arthur Godfrey 52 Toronto , Ontario , Canada Southampton Toronto , Ontario , Canada 6 Porter , Mr. Walter Chamberlain 46 Worcester , Massachusetts , US Southampton Worcester , Massachusetts , US 207 Potter , Mrs. Lily Alexenia ( née Wilson ) 56 Philadelphia , Pennsylvania , US Cherbourg Philadelphia , Pennsylvania , US 7 Reuchlin , The Honourable Mr. Johan George 38 Rotterdam , The Netherlands Southampton New York City Rheims , Mr. George Alexander Lucien 36 Paris , France Southampton New York City Robert , Mrs. Elisabeth Walton ( née McMillan ) 43 St. Louis , Missouri , US Southampton St. Louis , Missouri , US and maid , Miss Emilie Kreuchen 29 St. Louis , Missouri , US Southampton St. Louis , Missouri , US Roebling , Mr. Washington Augustus II 31 Trenton , New Jersey , US Southampton Trenton , New Jersey , US Romaine , Mr. Charles Hallance ( alias C. Rolmane ) 45 New York City Southampton New York City 9 Rood , Mr. Hugh Roscoe 38 Seattle , Washington , US Southampton Seattle , Washington , US Rosenbaum , Miss Edith Louise 34 Paris , France Cherbourg Cincinnati , Ohio , US 11 Rosenshine , Mr. George ( alias George Thorne ) 46 New York City Cherbourg New York City 16 Ross , Mr. John Hugo 36 Winnipeg , Manitoba , Canada Southampton Winnipeg , Manitoba , Canada Rothschild , Mr. Martin 46 New York City Cherbourg New York City Rothschild , Mrs. Elizabeth Jane Anne ( née Barrett ) 54 New York City Cherbourg New York City 6 Rowe , Mr. Alfred G . 59 Liverpool , Merseyside , England , UK Southampton New York City 109 Ryerson , Mr. Arthur Larned 61 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Mrs. Emily Maria ( née Borie ) 48 Cooperstown , New York , US Cherbourg Cooperstown , New York , US and maid , Miss Victorine Chaudanson 36 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Miss Susan Parker `` Suzette '' 21 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Miss Emily Borie 18 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Ryerson , Master John Borie `` Jack '' 13 Cooperstown , New York , US Cherbourg Cooperstown , New York , US and governess , Miss Grace Scott Bowen 45 Cooperstown , New York , US Cherbourg Cooperstown , New York , US Saalfeld , Mr. Adolphe 47 Manchester , England , UK Southampton New York City Salomon , Mr. Abraham Lincoln 43 New York City Cherbourg New York City Schabert , Mrs. Emma ( née Mock ) 35 Hamburg , German Empire Southampton New York City 11 Seward , Mr. Frederic Kimber 34 New York City Southampton New York City 7 Silverthorne , Mr. Spencer Victor 35 St. Louis , Missouri , US Southampton St. Louis , Missouri , US 5 Silvey , Mr. William Baird 50 Duluth , Minnesota , US Cherbourg Duluth , Minnesota , US Silvey , Mrs. Alice ( née Munger ) 39 Duluth , Minnesota , US Cherbourg Duluth , Minnesota , US 11 Simonius - Blumer , Colonel Alfons 56 Basel , Switzerland Southampton New York City Sloper , Mr. William Thompson 28 New Britain , Connecticut , US Southampton New Britain , Connecticut , US 7 Smart , Mr. John Montgomery 56 Kildale , North Yorkshire , England , UK Southampton New York City Smith , Mr. James Clinch 56 Paris , France Cherbourg Long Island , New York , US Smith , Mr. Lucien Philip 24 Huntington , West Virginia , US Cherbourg Huntington , West Virginia , US Smith , Mrs. Mary Eloise ( née Hughes ) 18 Huntington , West Virginia , US Cherbourg Huntington , West Virginia , US 6 Smith , Mr. Richard William 57 Streatham , London , England , UK Southampton New York City Snyder , Mr. John Pillsbury 24 Minneapolis , Minnesota , US Southampton Minneapolis , Minnesota , US 7 Snyder , Mrs. Nellie ( née Stevenson ) 23 Minneapolis , Minnesota , US Southampton Minneapolis , Minnesota , US 7 Spedden , Mr. Frederic Oakley 45 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spedden , Mrs. Margaretta Corning ( née Stone ) 39 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US and maid , Miss Helen Alice Wilson 31 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spedden , Master Robert Douglas 6 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US and nurse , Miss Elizabeth Margaret Burns 41 Tuxedo Park , New York , US Cherbourg Tuxedo Park , New York , US Spencer , Mr. William Augustus 57 New York City Cherbourg New York City Spencer , Mrs. Marie Eugénie ( née Demougeot ) 45 New York City Cherbourg New York City 6 and maid , Miss Eugenie Elise Lurette 59 New York City Cherbourg New York City 6 Stähelin - Maeglin , Dr. Max 32 Basel , Switzerland Southampton New York City Stead , Mr. William Thomas 62 London , England , UK Southampton New York City Stengel , Mr. Charles Emil Henry 54 Newark , New Jersey , US Cherbourg Newark , New Jersey , US Stengel , Mrs. Annie May ( née Morris ) 44 Newark , New Jersey , US Cherbourg Newark , New Jersey , US 5 Stephenson , Mrs. Martha ( née Eustis ) 52 Paris , France Cherbourg Haverford , Pennsylvania , US Stewart , Mr. Albert A . 54 Cincinnati , Ohio , US Cherbourg Gallipolis , Ohio , US Stone , Mrs. Martha Evelyn ( née Stevens ) 62 New York City Southampton New York City 6 and maid , Miss Amelie `` Amelia '' Icard 38 New York City Southampton New York City 6 Straus , Mr. Isidor 67 New York City Southampton New York City 96 and valet , Mr. John Farthing 57 New York City Southampton New York City Straus , Mrs. Rosalie Ida ( née Blun ) 63 New York City Southampton New York City and maid , Miss Ellen Bird 31 New York City Southampton New York City 8 Sutton , Mr. Frederick 61 Haddonfield , New Jersey , US Southampton Haddonfield , New Jersey , US 46 Swift , Mrs. Margaret Welles ( née Barron ) 46 New York City Southampton New York City 8 Taussig , Mr. Emil 52 New York City Southampton New York City Taussig , Mrs. Tillie ( née Mandelbaum ) 39 New York City Southampton New York City 8 Taussig , Miss Ruth 18 New York City Southampton New York City 8 Taylor , Mr. Elmer Zebley 48 London , England , UK Southampton East Orange , New Jersey , US 5 Taylor , Mrs. Juliet Cummins ( née Wright ) 48 London , England , UK Southampton East Orange , New Jersey , US 5 Thayer , Mr. John Borland II 49 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thayer , Mrs. Marian Longsteth ( née Morris ) 39 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US and maid , Miss Margaret Fleming 42 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thayer , Mr. John Borland `` Jack '' III 17 Haverford , Pennsylvania , US Cherbourg Haverford , Pennsylvania , US Thorne , Mrs. Gertrude Maybelle ( née McMinn ) 37 New York City , New York , US Cherbourg New York City , New York , US Tucker , Mr. Gilbert Milligan Jr . 31 Albany , New York , US Cherbourg Albany , New York , US 7 Uruchurtu , Don Manuel E . 40 Mexico City , Mexico Cherbourg Mexico City , Mexico Van der Hoef , Mr. Wyckoff 61 Brooklyn , New York , US Belfast Brooklyn , New York , US 245 Walker , Mr. William Anderson 48 East Orange , New Jersey , US Southampton East Orange , New Jersey , US Warren , Mr. Frank Manley 63 Portland , Oregon , US Cherbourg Portland , Oregon , US Warren , Mrs. Anna Sophia ( née Atkinson ) 60 Portland , Oregon , US Cherbourg Portland , Oregon , US 5 Weir , Colonel John 59 New York City Southampton New York City White , Mr. Percival Wayland 54 Brunswick , Maine , US Southampton Brunswick , Maine , US White , Mr. Richard Frasar 21 Brunswick , Maine , US Southampton Brunswick , Maine , US 169 White , Mrs. Ella ( née Holmes ) 55 New York City Cherbourg New York City 8 and maid , Miss Amelia Mayo `` Nellie '' Bessette 39 New York City Cherbourg New York City 8 and manservant , Mr. Sante Righini 22 New York City Cherbourg New York City 232 Wick , Colonel George Dennick 58 Youngstown , Ohio , US Southampton Youngstown , Ohio , US Wick , Mrs. Mary ( née Hitchcock ) 45 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Wick , Miss Mary Natalie 31 Youngstown , Ohio , US Southampton Youngstown , Ohio , US 8 Widener , Mr. George Dunton 50 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US and valet , Mr. Edwin Herbert Keeping 33 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US 45 Widener , Mrs. Eleanor ( née Elkins ) 50 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US and maid , Miss Amalie Henriette `` Emily '' Gieger 35 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Widener , Mr. Harry Elkins 27 Philadelphia , Pennsylvania , US Southampton Philadelphia , Pennsylvania , US Willard , Miss Constance 21 Duluth , Minnesota , US Southampton Duluth , Minnesota , US 8 Williams , Mr. Charles Duane 51 Geneva , Switzerland Cherbourg Radnor , Pennsylvania , US Williams , Mr. Richard Norris II 21 Geneva , Switzerland Cherbourg Radnor , Pennsylvania , US Woolner , Mr. Hugh 45 London , England , UK Southampton New York City Wright , Mr. George 62 Halifax , Nova Scotia , Canada Southampton Halifax , Nova Scotia , Canada Young , Miss Marie Grice 36 New York City Cherbourg New York City 8
  Fact 0.0: Allen, Miss Elizabeth Walton was 29 years old.
  Fact 0.1: Allen, Miss Elizabeth Walton is from St. Louis, Missouri, US.
  Fact 0.2: Allen, Miss Elizabeth Walton boarded at Southampton.
  Fact 0.3: Allen, Miss Elizabeth Walton's destination is St. Louis, Missouri, US.
  Fact 0.4: Allison, Mr. Hudson Joshua Creighton was 30 years old.
  Fact 0.5: Allison, Mr. Hudson Joshua Creighton is from Montreal, Quebec, Canada.
  Fact 0.6: Allison, Mr. Hudson Joshua Creighton boarded at Southampton.
  Fact 0.7: Allison, Mr. Hudson Joshua Creighton's destination is Montreal, Quebec, Canada.
  Fact 0.8: Allison, Mr. Hudson Joshua Creighton has 135 in the lifeboat or body column.
  Fact 0.9: Mr. George Swane was 19 years old.
  Fact 0.10: Mr. George Swane is listed as the chauffeur.
  Fact 0.11: Mr. George Swane is from Montreal, Quebec, Canada.
  Fact 0.12: Mr. George Swane boarded at Southampton.
  Fact 0.13: Mr. George Swane's destination is Montreal, Quebec, Canada.
  Fact 0.14: Mr. George Swane has 294 in the lifeboat or body column.
  Fact 0.15: Miss Amelia Mary "Mildred" Brown was 18 years old.
  Fact 0.16: Miss Amelia Mary "Mildred" Brown is listed as the cook.
  Fact 0.17: Miss Amelia Mary "Mildred" Brown is from London, England, UK.
  Fact 0.18: Miss Amelia Mary "Mildred" Brown boarded at Southampton.
  Fact 0.19: Miss Amelia Mary "Mildred" Brown's destination is Montreal, Quebec, Canada.
  Fact 0.20: Miss Amelia Mary "Mildred" Brown has 11 in the lifeboat or body column.
  Fact 0.21: Allison, Mrs. Bessie Waldo (née Daniels) was 25 years old.
  Fact 0.22: Allison, Mrs. Bessie Waldo (née Daniels) is from Montreal, Quebec, Canada.
  Fact 0.23: Allison, Mrs. Bessie Waldo (née Daniels) boarded at Southampton.
  Fact 0.24: Allison, Mrs. Bessie Waldo (née Daniels)'s destination is Montreal, Quebec, Canada.
  Fact 0.25: Miss Sarah Daniels was 33 years old.
  Fact 0.26: Miss Sarah Daniels is listed as the maid.
  Fact 0.27: Miss Sarah Daniels is from Montreal, Quebec, Canada.
  Fact 0.28: Miss Sarah Daniels boarded at Southampton.
  Fact 0.29: Miss Sarah Daniels's destination is Montreal, Quebec, Canada.
  Fact 0.30: Miss Sarah Daniels has 8 in the lifeboat or body column.
  Fact 0.31: Allison, Miss Helen Loraine is from Montreal, Quebec, Canada.
  Fact 0.32: Allison, Miss Helen Loraine boarded at Southampton.
  Fact 0.33: Allison, Miss Helen Loraine's destination is Montreal, Quebec, Canada.
  Fact 0.34: Allison, Master Hudson Trevor is listed as age 0.11 and as 11 months.
  Fact 0.35: Allison, Master Hudson Trevor is from Montreal, Quebec, Canada.
  Fact 0.36: Allison, Master Hudson Trevor boarded at Southampton.
  Fact 0.37: Allison, Master Hudson Trevor's destination is Montreal, Quebec, Canada.
  Fact 0.38: Allison, Master Hudson Trevor has 11 in the lifeboat or body column.
  Fact 0.39: Miss Alice Catherine Cleaver was 22 years old.
  Fact 0.40: Miss Alice Catherine Cleaver is listed as the nurse.
  Fact 0.41: Miss Alice Catherine Cleaver is from London, England, UK.
  Fact 0.42: Miss Alice Catherine Cleaver boarded at Southampton.
  Fact 0.43: Miss Alice Catherine Cleaver's destination is Montreal, Quebec, Canada.
  Fact 0.44: Miss Alice Catherine Cleaver has 11 in the lifeboat or body column.
  Fact 0.45: Anderson, Mr. Harry was 47 years old.
  Fact 0.46: Anderson, Mr. Harry is from New York City, New York, US.
  Fact 0.47: Anderson, Mr. Harry boarded at Southampton.
  Fact 0.48: Anderson, Mr. Harry's destination is New York City, New York, US.
  Fact 0.49: Andrews, Miss Kornelia Theodosia was 62 years old.
  Fact 0.50: Andrews, Miss Kornelia Theodosia is from Hudson, New York, US.
  Fact 0.51: Andrews, Miss Kornelia Theodosia boarded at Cherbourg.
  Fact 0.52: Andrews, Miss Kornelia Theodosia's destination is Hudson, New York, US.
  Fact 0.53: Andrews, Miss Kornelia Theodosia has 10 in the lifeboat or body column.
  Fact 0.54: Andrews, Mr. Thomas was 39 years old.
  Fact 0.55: Andrews, Mr. Thomas is from Belfast, Ireland, UK.
  Fact 0.56: Andrews, Mr. Thomas boarded at Belfast.
  Fact 0.57: Andrews, Mr. Thomas's destination is New York City.
  Fact 0.58: Appleton, Mrs. Charlotte Lane (née Lamson) was 53 years old.
  Fact 0.59: Appleton, Mrs. Charlotte Lane (née Lamson) is from New York City.
  Fact 0.60: Appleton, Mrs. Charlotte Lane (née Lamson) boarded at Southampton.
  Fact 0.61: Appleton, Mrs. Charlotte Lane (née Lamson)'s destination is New York City.
  Fact 0.62: Artagaveytia, Mr. Ramon was 71 years old.
  Fact 0.63: Artagaveytia, Mr. Ramon is from Buenos Aires, Argentina.
  Fact 0.64: Artagaveytia, Mr. Ramon boarded at Cherbourg.
  Fact 0.65: Artagaveytia, Mr. Ramon's destination is New York City.
  Fact 0.66: Artagaveytia, Mr. Ramon has 22 in the lifeboat or body column.
  Fact 0.67: Astor, Colonel John Jacob IV was 47 years old.
  Fact 0.68: Astor, Colonel John Jacob IV is from New York City.
  Fact 0.69: Astor, Colonel John Jacob IV boarded at Cherbourg.
  Fact 0.70: Astor, Colonel John Jacob IV's destination is New York City.
  Fact 0.71: Astor, Colonel John Jacob IV has 124 in the lifeboat or body column.
  Fact 0.72: Astor, Mrs. Madeleine Talmage (née Force) was 18 years old.
  Fact 0.73: Astor, Mrs. Madeleine Talmage (née Force) is from New York City.
  Fact 0.74: Astor, Mrs. Madeleine Talmage (née Force) boarded at Cherbourg.
  Fact 0.75: Astor, Mrs. Madeleine Talmage (née Force)'s destination is New York City.
  Fact 0.76: Miss Rosalie Bidois was 46 years old.
  Fact 0.77: Miss Rosalie Bidois is listed as the maid.
  Fact 0.78: Miss Rosalie Bidois is from New York City.
  Fact 0.79: Miss Rosalie Bidois boarded at Cherbourg.
  Fact 0.80: Miss Rosalie Bidois's destination is New York City.
  Fact 0.81: Miss Caroline Louise Endres was 39 years old.
  Fact 0.82: Miss Caroline Louise Endres is listed as the nurse.
  Fact 0.83: Miss Caroline Louise Endres is from New York City.
  Fact 0.84: Miss Caroline Louise Endres boarded at Cherbourg.
  Fact 0.85: Miss Caroline Louise Endres's destination is New York City.
  Fact 0.86: Aubart, Mrs. Léontine Pauline was 24 years old.
  Fact 0.87: Aubart, Mrs. Léontine Pauline is from Paris, France.
  Fact 0.88: Aubart, Mrs. Léontine Pauline boarded at Cherbourg.
  Fact 0.89: Aubart, Mrs. Léontine Pauline's destination is New York City.
  Fact 0.90: Aubart, Mrs. Léontine Pauline has 9 in the lifeboat or body column.
  Fact 0.91: Miss Emma Sägesser was 24 years old.
  Fact 0.92: Miss Emma Sägesser is listed as the maid.
  Fact 0.93: Miss Emma Sägesser is from Paris, France.
  Fact 0.94: Miss Emma Sägesser boarded at Cherbourg.
  Fact 0.95: Miss Emma Sägesser's destination is New York City.
  Fact 0.96: Miss Emma Sägesser has 9 in the lifeboat or body column.
  Fact 0.97: Barkworth, Mr. Algernon Henry Wilson was 47 years old.
  Fact 0.98: Barkworth, Mr. Algernon Henry Wilson is from Hessle, East Yorkshire, England, UK.
  Fact 0.99: Barkworth, Mr. Algernon Henry Wilson boarded at Southampton.
  Fact 0.100: Barkworth, Mr. Algernon Henry Wilson's destination is New York City.
  Fact 0.101: Baumann, Mr. John D. was 60 years old.
  Fact 0.102: Baumann, Mr. John D. is from New York City.
  Fact 0.103: Baumann, Mr. John D. boarded at Cherbourg.
  Fact 0.104: Baumann, Mr. John D.'s destination is New York City.
  Fact 0.105: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) was 50 years old.
  Fact 0.106: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) is from Montreal, Quebec, Canada.
  Fact 0.107: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) boarded at Cherbourg.
  Fact 0.108: Baxter, Mrs. Hélène (née de Lanaudière-Chaput)'s destination is Montreal, Quebec, Canada.
  Fact 0.109: Baxter, Mrs. Hélène (née de Lanaudière-Chaput) has 6 in the lifeboat or body column.
  Fact 0.110: Baxter, Mr. Quigg Edmond was 24 years old.
  Fact 0.111: Baxter, Mr. Quigg Edmond is from Montreal, Quebec, Canada.
  Fact 0.112: Baxter, Mr. Quigg Edmond boarded at Cherbourg.
  Fact 0.113: Baxter, Mr. Quigg Edmond's destination is Montreal, Quebec, Canada.
  Fact 0.114: Beattie, Mr. Thomson was 36 years old.
  Fact 0.115: Beattie, Mr. Thomson is from Fergus, Ontario, Canada.
  Fact 0.116: Beattie, Mr. Thomson boarded at Southampton.
  Fact 0.117: Beattie, Mr. Thomson's destination is Fergus, Ontario, Canada.
  Fact 0.118: Beattie, Mr. Thomson has 331 in the lifeboat or body column.
  Fact 0.119: Beckwith, Mr. Richard Leonard was 37 years old.
  Fact 0.120: Beckwith, Mr. Richard Leonard is from New York City.
  Fact 0.121: Beckwith, Mr. Richard Leonard boarded at Southampton.
  Fact 0.122: Beckwith, Mr. Richard Leonard's destination is New York City.
  Fact 0.123: Beckwith, Mr. Richard Leonard has 5 in the lifeboat or body column.
  Fact 0.124: Beckwith, Mrs. Sallie (née Monypeny) was 46 years old.
  Fact 0.125: Beckwith, Mrs. Sallie (née Monypeny) is from New York City.
  Fact 0.126: Beckwith, Mrs. Sallie (née Monypeny) boarded at Southampton.
  Fact 0.127: Beckwith, Mrs. Sallie (née Monypeny)'s destination is New York City.
  Fact 0.128: Beckwith, Mrs. Sallie (née Monypeny) has 5 in the lifeboat or body column.
  Fact 0.129: Behr, Mr. Karl Howell was 26 years old.
  Fact 0.130: Behr, Mr. Karl Howell is from New York City.
  Fact 0.131: Behr, Mr. Karl Howell boarded at Cherbourg.
  Fact 0.132: Behr, Mr. Karl Howell's destination is New York City.
  Fact 0.133: Behr, Mr. Karl Howell has 5 in the lifeboat or body column.
  Fact 0.134: Birnbaum, Mr. Jakob was 24 years old.
  Fact 0.135: Birnbaum, Mr. Jakob is from Antwerp, Belgium.
  Fact 0.136: Birnbaum, Mr. Jakob boarded at Cherbourg.
  Fact 0.137: Birnbaum, Mr. Jakob's destination is San Francisco, California, US.
  Fact 0.138: Birnbaum, Mr. Jakob has 148 in the lifeboat or body column.
  Fact 0.139: Bishop, Mr. Dickinson H. "Dick" was 25 years old.
  Fact 0.140: Bishop, Mr. Dickinson H. "Dick" is from Dowagiac, Michigan, US.
  Fact 0.141: Bishop, Mr. Dickinson H. "Dick" boarded at Cherbourg.
  Fact 0.142: Bishop, Mr. Dickinson H. "Dick"'s destination is Dowagiac, Michigan, US.
  Fact 0.143: Bishop, Mr. Dickinson H. "Dick" has 7 in the lifeboat or body column.
  Fact 0.144: Bishop, Mrs. Helen (née Walton) was 19 years old.
  Fact 0.145: Bishop, Mrs. Helen (née Walton) is from Dowagiac, Michigan, US.
  Fact 0.146: Bishop, Mrs. Helen (née Walton) boarded at Cherbourg.
  Fact 0.147: Bishop, Mrs. Helen (née Walton)'s destination is Dowagiac, Michigan, US.
  Fact 0.148: Bishop, Mrs. Helen (née Walton) has 7 in the lifeboat or body column.
  Fact 0.149: Björnström-Steffanson, Mr. Mauritz Håkan was 28 years old.
  Fact 0.150: Björnström-Steffanson, Mr. Mauritz Håkan is from Stockholm, Sweden.
  Fact 0.151: Björnström-Steffanson, Mr. Mauritz Håkan boarded at Southampton.
  Fact 0.152: Björnström-Steffanson, Mr. Mauritz Håkan's destination is Washington, D.C., US.
  Fact 0.153: Blackwell, Mr. Stephen Weart was 45 years old.
  Fact 0.154: Blackwell, Mr. Stephen Weart is from Trenton, New Jersey, US.
  Fact 0.155: Blackwell, Mr. Stephen Weart boarded at Southampton.
  Fact 0.156: Blackwell, Mr. Stephen Weart's destination is Trenton, New Jersey, US.
  Fact 0.157: Blank, Mr. Henry was 39 years old.
  Fact 0.158: Blank, Mr. Henry is from Glen Ridge, New Jersey, US.
  Fact 0.159: Blank, Mr. Henry boarded at Cherbourg.
  Fact 0.160: Blank, Mr. Henry's destination is Glen Ridge, New Jersey, US.
  Fact 0.161: Blank, Mr. Henry has 7 in the lifeboat or body column.
  Fact 0.162: Bonnell, Miss Elizabeth was 61 years old.
  Fact 0.163: Bonnell, Miss Elizabeth is from Youngstown, Ohio, US.
  Fact 0.164: Bonnell, Miss Elizabeth boarded at Southampton.
  Fact 0.165: Bonnell, Miss Elizabeth's destination is Youngstown, Ohio, US.
  Fact 0.166: Bonnell, Miss Elizabeth has 8 in the lifeboat or body column.
  Fact 0.167: Bonnell, Miss Caroline was 30 years old.
  Fact 0.168: Bonnell, Miss Caroline is from Youngstown, Ohio, US.
  Fact 0.169: Bonnell, Miss Caroline boarded at Southampton.
  Fact 0.170: Bonnell, Miss Caroline's destination is Youngstown, Ohio, US.
  Fact 0.171: Bonnell, Miss Caroline has 8 in the lifeboat or body column.
  Fact 0.172: Borebank, Mr. John James was 42 years old.
  Fact 0.173: Borebank, Mr. John James is from London, England, UK.
  Fact 0.174: Borebank, Mr. John James boarded at Southampton.
  Fact 0.175: Borebank, Mr. John James's destination is Toronto, Ontario, Canada.
  Fact 0.176: Bowerman, Miss Elsie Edith was 22 years old.
  Fact 0.177: Bowerman, Miss Elsie Edith is from St. Leonards-on-Sea, East Sussex, England, UK.
  Fact 0.178: Bowerman, Miss Elsie Edith boarded at Southampton.
  Fact 0.179: Bowerman, Miss Elsie Edith's destination is New York City.
  Fact 0.180: Bowerman, Miss Elsie Edith has 6 in the lifeboat or body column.
  Fact 0.181: Brady, Mr. John Bertram was 41 years old.
  Fact 0.182: Brady, Mr. John Bertram is from Pomeroy, Washington, US.
  Fact 0.183: Brady, Mr. John Bertram boarded at Southampton.
  Fact 0.184: Brady, Mr. John Bertram's destination is Pomeroy, Washington, US.
  Fact 0.185: Brandeis, Mr. Emil was 48 years old.
  Fact 0.186: Brandeis, Mr. Emil is from Omaha, Nebraska, US.
  Fact 0.187: Brandeis, Mr. Emil boarded at Cherbourg.
  Fact 0.188: Brandeis, Mr. Emil's destination is Omaha, Nebraska, US.
  Fact 0.189: Brandeis, Mr. Emil has 208 in the lifeboat or body column.
  Fact 0.190: Brereton, Mr. George Andrew (alias George A. Brayton) was 37 years old.
  Fact 0.191: Brereton, Mr. George Andrew (alias George A. Brayton) is from Los Angeles, US.
  Fact 0.192: Brereton, Mr. George Andrew (alias George A. Brayton) boarded at Southampton.
  Fact 0.193: Brereton, Mr. George Andrew (alias George A. Brayton)'s destination is Los Angeles, California, US.
  Fact 0.194: Brereton, Mr. George Andrew (alias George A. Brayton) has 9 in the lifeboat or body column.
  Fact 0.195: Brewe, Dr. Arthur Jackson was 45 years old.
  Fact 0.196: Brewe, Dr. Arthur Jackson is from Philadelphia, Pennsylvania, US.
  Fact 0.197: Brewe, Dr. Arthur Jackson boarded at Cherbourg.
  Fact 0.198: Brewe, Dr. Arthur Jackson's destination is Philadelphia, Pennsylvania, US.
  Fact 0.199: Brown, Mrs. Caroline Lane (née Lamson) was 59 years old.
  Fact 0.200: Brown, Mrs. Caroline Lane (née Lamson) is from Belmont, Massachusetts, US.
  Fact 0.201: Brown, Mrs. Caroline Lane (née Lamson) boarded at Southampton.
  Fact 0.202: Brown, Mrs. Caroline Lane (née Lamson)'s destination is Belmont, Massachusetts, US.
  Fact 0.203: Brown, Mrs. Margaret (née Tobin) was 44 years old.
  Fact 0.204: Brown, Mrs. Margaret (née Tobin) is from Denver, Colorado, US.
  Fact 0.205: Brown, Mrs. Margaret (née Tobin) boarded at Cherbourg.
  Fact 0.206: Brown, Mrs. Margaret (née Tobin)'s destination is Denver, Colorado, US.
  Fact 0.207: Brown, Mrs. Margaret (née Tobin) has 6 in the lifeboat or body column.
  Fact 0.208: Bucknell, Mrs. Emma Eliza (née Ward) was 59 years old.
  Fact 0.209: Bucknell, Mrs. Emma Eliza (née Ward) is from Philadelphia, Pennsylvania, US.
  Fact 0.210: Bucknell, Mrs. Emma Eliza (née Ward) boarded at Cherbourg.
  Fact 0.211: Bucknell, Mrs. Emma Eliza (née Ward)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.212: Bucknell, Mrs. Emma Eliza (née Ward) has 8 in the lifeboat or body column.
  Fact 0.213: Miss Albina Bazzani was 36 years old.
  Fact 0.214: Miss Albina Bazzani is listed as the maid.
  Fact 0.215: Miss Albina Bazzani is from Philadelphia, Pennsylvania, US.
  Fact 0.216: Miss Albina Bazzani boarded at Cherbourg.
  Fact 0.217: Miss Albina Bazzani's destination is Philadelphia, Pennsylvania, US.
  Fact 0.218: Miss Albina Bazzani has 8 in the lifeboat or body column.
  Fact 0.219: Butt, Major Archibald Willingham was 46 years old.
  Fact 0.220: Butt, Major Archibald Willingham is from Washington, D.C., US.
  Fact 0.221: Butt, Major Archibald Willingham boarded at Southampton.
  Fact 0.222: Butt, Major Archibald Willingham's destination is Washington, D.C., US.
  Fact 0.223: Calderhead, Mr. Edward Pennington was 42 years old.
  Fact 0.224: Calderhead, Mr. Edward Pennington is from New York City.
  Fact 0.225: Calderhead, Mr. Edward Pennington boarded at Southampton.
  Fact 0.226: Calderhead, Mr. Edward Pennington's destination is New York City.
  Fact 0.227: Calderhead, Mr. Edward Pennington has 5 in the lifeboat or body column.
  Fact 0.228: Candee, Mrs. Helen Churchill (née Hungerford) was 53 years old.
  Fact 0.229: Candee, Mrs. Helen Churchill (née Hungerford) is from Washington, D.C., US.
  Fact 0.230: Candee, Mrs. Helen Churchill (née Hungerford) boarded at Cherbourg.
  Fact 0.231: Candee, Mrs. Helen Churchill (née Hungerford)'s destination is Washington, D.C., US.
  Fact 0.232: Candee, Mrs. Helen Churchill (née Hungerford) has 6 in the lifeboat or body column.
  Fact 0.233: Cardeza, Mrs. Charlotte Wardle (née Drake) was 58 years old.
  Fact 0.234: Cardeza, Mrs. Charlotte Wardle (née Drake) is from Germantown, Pennsylvania, US.
  Fact 0.235: Cardeza, Mrs. Charlotte Wardle (née Drake) boarded at Cherbourg.
  Fact 0.236: Cardeza, Mrs. Charlotte Wardle (née Drake)'s destination is Germantown, Pennsylvania, US.
  Fact 0.237: Miss Annie Moore Ward was 38 years old.
  Fact 0.238: Miss Annie Moore Ward is listed as the maid.
  Fact 0.239: Miss Annie Moore Ward is from Germantown, Pennsylvania, US.
  Fact 0.240: Miss Annie Moore Ward boarded at Cherbourg.
  Fact 0.241: Miss Annie Moore Ward's destination is Germantown, Pennsylvania, US.
  Fact 0.242: Cardeza, Mr. Thomas Drake Martinez was 36 years old.
  Fact 0.243: Cardeza, Mr. Thomas Drake Martinez is from Germantown, Pennsylvania, US.
  Fact 0.244: Cardeza, Mr. Thomas Drake Martinez boarded at Cherbourg.
  Fact 0.245: Cardeza, Mr. Thomas Drake Martinez's destination is Germantown, Pennsylvania, US.
  Fact 0.246: Mr. Gustave J. Lesueur was 35 years old.
  Fact 0.247: Mr. Gustave J. Lesueur is listed as the valet.
  Fact 0.248: Mr. Gustave J. Lesueur is from Germantown, Pennsylvania, US.
  Fact 0.249: Mr. Gustave J. Lesueur boarded at Cherbourg.
  Fact 0.250: Mr. Gustave J. Lesueur's destination is Germantown, Pennsylvania, US.
  Fact 0.251: Carlsson, Mr. Frans Olof was 33 years old.
  Fact 0.252: Carlsson, Mr. Frans Olof is from New York City.
  Fact 0.253: Carlsson, Mr. Frans Olof boarded at Southampton.
  Fact 0.254: Carlsson, Mr. Frans Olof's destination is New York City.
  Fact 0.255: Carrau, Mr. Francisco M. was 31 years old.
  Fact 0.256: Carrau, Mr. Francisco M. is from Montevideo, Uruguay.
  Fact 0.257: Carrau, Mr. Francisco M. boarded at Southampton.
  Fact 0.258: Carrau, Mr. Francisco M.'s destination is Montevideo, Uruguay.
  Fact 0.259: Carrau, Mr. José Pedro was 17 years old.
  Fact 0.260: Carrau, Mr. José Pedro is from Montevideo, Uruguay.
  Fact 0.261: Carrau, Mr. José Pedro boarded at Southampton.
  Fact 0.262: Carrau, Mr. José Pedro's destination is Montevideo, Uruguay.
  Fact 0.263: Carter, Mr. William Ernest was 36 years old.
  Fact 0.264: Carter, Mr. William Ernest is from Bryn Mawr, Pennsylvania, US.
  Fact 0.265: Carter, Mr. William Ernest boarded at Southampton.
  Fact 0.266: Carter, Mr. William Ernest's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.267: Mr. Alexander Cairns was 28 years old.
  Fact 0.268: Mr. Alexander Cairns is listed as the valet.
  Fact 0.269: Mr. Alexander Cairns is from Bryn Mawr, Pennsylvania, US.
  Fact 0.270: Mr. Alexander Cairns boarded at Southampton.
  Fact 0.271: Mr. Alexander Cairns's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.272: Mr. Charles Augustus Aldworth was 30 years old.
  Fact 0.273: Mr. Charles Augustus Aldworth is listed as the chauffeur.
  Fact 0.274: Mr. Charles Augustus Aldworth is from Bryn Mawr, Pennsylvania, US.
  Fact 0.275: Mr. Charles Augustus Aldworth boarded at Southampton.
  Fact 0.276: Mr. Charles Augustus Aldworth's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.277: Carter, Mrs. Lucile (née Polk) was 36 years old.
  Fact 0.278: Carter, Mrs. Lucile (née Polk) is from Bryn Mawr, Pennsylvania, US.
  Fact 0.279: Carter, Mrs. Lucile (née Polk) boarded at Southampton.
  Fact 0.280: Carter, Mrs. Lucile (née Polk)'s destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.281: Miss Auguste Serreplan was 30 years old.
  Fact 0.282: Miss Auguste Serreplan is listed as the maid.
  Fact 0.283: Miss Auguste Serreplan is from Bryn Mawr, Pennsylvania, US.
  Fact 0.284: Miss Auguste Serreplan boarded at Southampton.
  Fact 0.285: Miss Auguste Serreplan's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.286: Carter, Miss Lucile Polk was 13 years old.
  Fact 0.287: Carter, Miss Lucile Polk is from Bryn Mawr, Pennsylvania, US.
  Fact 0.288: Carter, Miss Lucile Polk boarded at Southampton.
  Fact 0.289: Carter, Miss Lucile Polk's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.290: Carter, Master William Thornton II was 11 years old.
  Fact 0.291: Carter, Master William Thornton II is from Bryn Mawr, Pennsylvania, US.
  Fact 0.292: Carter, Master William Thornton II boarded at Southampton.
  Fact 0.293: Carter, Master William Thornton II's destination is Bryn Mawr, Pennsylvania, US.
  Fact 0.294: Case, Mr. Howard Brown was 49 years old.
  Fact 0.295: Case, Mr. Howard Brown is from Ascot, Berkshire, England, UK.
  Fact 0.296: Case, Mr. Howard Brown boarded at Southampton.
  Fact 0.297: Case, Mr. Howard Brown's destination is Rochester, New York, US.
  Fact 0.298: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) was 36 years old.
  Fact 0.299: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) is from New York City.
  Fact 0.300: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) boarded at Cherbourg.
  Fact 0.301: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick)'s destination is New York City.
  Fact 0.302: Cassebeer, Mrs. Eleanor Genevieve (née Fosdick) has 5 in the lifeboat or body column.
  Fact 0.303: Cavendish, Mr. Tyrell William was 36 years old.
  Fact 0.304: Cavendish, Mr. Tyrell William is from London, England, UK.
  Fact 0.305: Cavendish, Mr. Tyrell William boarded at Southampton.
  Fact 0.306: Cavendish, Mr. Tyrell William's destination is New York City.
  Fact 0.307: Cavendish, Mr. Tyrell William has 172 in the lifeboat or body column.
  Fact 0.308: Cavendish, Mrs. Julia Florence (née Siegel) was 25 years old.
  Fact 0.309: Cavendish, Mrs. Julia Florence (née Siegel) is from London, England, UK.
  Fact 0.310: Cavendish, Mrs. Julia Florence (née Siegel) boarded at Southampton.
  Fact 0.311: Cavendish, Mrs. Julia Florence (née Siegel)'s destination is New York City.
  Fact 0.312: Cavendish, Mrs. Julia Florence (née Siegel) has 6 in the lifeboat or body column.
  Fact 0.313: Miss Ellen "Nellie" Barber was 26 years old.
  Fact 0.314: Miss Ellen "Nellie" Barber is listed as the maid.
  Fact 0.315: Miss Ellen "Nellie" Barber is from London, England, UK.
  Fact 0.316: Miss Ellen "Nellie" Barber boarded at Southampton.
  Fact 0.317: Miss Ellen "Nellie" Barber's destination is New York City.
  Fact 0.318: Miss Ellen "Nellie" Barber has 6 in the lifeboat or body column.
  Fact 0.319: Chaffee, Mr. Hubert Fuller was 46 years old.
  Fact 0.320: Chaffee, Mr. Hubert Fuller is from Amenia, North Dakota, US.
  Fact 0.321: Chaffee, Mr. Hubert Fuller boarded at Southampton.
  Fact 0.322: Chaffee, Mr. Hubert Fuller's destination is Amenia, North Dakota, US.
  Fact 0.323: Chaffee, Mrs. Carrie Constance (née Toogood) was 47 years old.
  Fact 0.324: Chaffee, Mrs. Carrie Constance (née Toogood) is from Amenia, North Dakota, US.
  Fact 0.325: Chaffee, Mrs. Carrie Constance (née Toogood) boarded at Southampton.
  Fact 0.326: Chaffee, Mrs. Carrie Constance (née Toogood)'s destination is Amenia, North Dakota, US.
  Fact 0.327: Chambers, Mr. Norman Campbell was 27 years old.
  Fact 0.328: Chambers, Mr. Norman Campbell is from New York City.
  Fact 0.329: Chambers, Mr. Norman Campbell boarded at Southampton.
  Fact 0.330: Chambers, Mr. Norman Campbell's destination is New York City.
  Fact 0.331: Chambers, Mr. Norman Campbell has 5 in the lifeboat or body column.
  Fact 0.332: Chambers, Mrs. Bertha (née Griggs) was 32 years old.
  Fact 0.333: Chambers, Mrs. Bertha (née Griggs) is from New York City.
  Fact 0.334: Chambers, Mrs. Bertha (née Griggs) boarded at Southampton.
  Fact 0.335: Chambers, Mrs. Bertha (née Griggs)'s destination is New York City.
  Fact 0.336: Chambers, Mrs. Bertha (née Griggs) has 5 in the lifeboat or body column.
  Fact 0.337: Cherry, Miss Gladys was 30 years old.
  Fact 0.338: Cherry, Miss Gladys is from London, England, UK.
  Fact 0.339: Cherry, Miss Gladys boarded at Southampton.
  Fact 0.340: Cherry, Miss Gladys's destination is Vancouver, British Columbia, Canada.
  Fact 0.341: Cherry, Miss Gladys has 8 in the lifeboat or body column.
  Fact 0.342: Chevré, Mr. Paul Romaine Marie Léonce was 45 years old.
  Fact 0.343: Chevré, Mr. Paul Romaine Marie Léonce is from Paris, France.
  Fact 0.344: Chevré, Mr. Paul Romaine Marie Léonce boarded at Cherbourg.
  Fact 0.345: Chevré, Mr. Paul Romaine Marie Léonce's destination is Ottawa, Ontario, Canada.
  Fact 0.346: Chevré, Mr. Paul Romaine Marie Léonce has 7 in the lifeboat or body column.
  Fact 0.347: Chibnall, Mrs. Edith Martha Bowerman (née Barber) was 48 years old.
  Fact 0.348: Chibnall, Mrs. Edith Martha Bowerman (née Barber) is from St. Leonards-on-Sea, East Sussex, England, UK.
  Fact 0.349: Chibnall, Mrs. Edith Martha Bowerman (née Barber) boarded at Southampton.
  Fact 0.350: Chibnall, Mrs. Edith Martha Bowerman (née Barber)'s destination is New York City.
  Fact 0.351: Chibnall, Mrs. Edith Martha Bowerman (née Barber) has 6 in the lifeboat or body column.
  Fact 0.352: Chisholm, Mr. Roderick Robert Crispin was 40 years old.
  Fact 0.353: Chisholm, Mr. Roderick Robert Crispin is from Belfast, Ireland, UK.
  Fact 0.354: Chisholm, Mr. Roderick Robert Crispin boarded at Belfast.
  Fact 0.355: Chisholm, Mr. Roderick Robert Crispin's destination is New York City.
  Fact 0.356: Clark, Mr. Walter Miller was 27 years old.
  Fact 0.357: Clark, Mr. Walter Miller is from Los Angeles, California, US.
  Fact 0.358: Clark, Mr. Walter Miller boarded at Cherbourg.
  Fact 0.359: Clark, Mr. Walter Miller's destination is Los Angeles, California, US.
  Fact 0.360: Clark, Mrs. Virginia Estelle (née McDowell) was 26 years old.
  Fact 0.361: Clark, Mrs. Virginia Estelle (née McDowell) is from Los Angeles, California, US.
  Fact 0.362: Clark, Mrs. Virginia Estelle (née McDowell) boarded at Cherbourg.
  Fact 0.363: Clark, Mrs. Virginia Estelle (née McDowell)'s destination is Los Angeles, California, US.
  Fact 0.364: Clifford, Mr. George Quincy was 40 years old.
  Fact 0.365: Clifford, Mr. George Quincy is from Stoughton, Massachusetts, US.
  Fact 0.366: Clifford, Mr. George Quincy boarded at Southampton.
  Fact 0.367: Clifford, Mr. George Quincy's destination is Stoughton, Massachusetts.
  Fact 0.368: Colley, Mr. Edward Pomeroy was 37 years old.
  Fact 0.369: Colley, Mr. Edward Pomeroy is from Dublin, Ireland, UK.
  Fact 0.370: Colley, Mr. Edward Pomeroy boarded at Southampton.
  Fact 0.371: Colley, Mr. Edward Pomeroy's destination is Vancouver, British Columbia, Canada.
  Fact 0.372: Compton, Mrs. Mary Eliza (née Ingersoll) was 64 years old.
  Fact 0.373: Compton, Mrs. Mary Eliza (née Ingersoll) is from Lakewood, New Jersey, US.
  Fact 0.374: Compton, Mrs. Mary Eliza (née Ingersoll) boarded at Cherbourg.
  Fact 0.375: Compton, Mrs. Mary Eliza (née Ingersoll)'s destination is Lakewood, New Jersey, US.
  Fact 0.376: Compton, Mrs. Mary Eliza (née Ingersoll) has 14 in the lifeboat or body column.
  Fact 0.377: Compton, Miss Sara Rebecca was 39 years old.
  Fact 0.378: Compton, Miss Sara Rebecca is from Lakewood, New Jersey, US.
  Fact 0.379: Compton, Miss Sara Rebecca boarded at Cherbourg.
  Fact 0.380: Compton, Miss Sara Rebecca's destination is Lakewood, New Jersey, US.
  Fact 0.381: Compton, Miss Sara Rebecca has 14 in the lifeboat or body column.
  Fact 0.382: Compton, Mr. Alexander Taylor Jr. was 37 years old.
  Fact 0.383: Compton, Mr. Alexander Taylor Jr. is from Lakewood, New Jersey, US.
  Fact 0.384: Compton, Mr. Alexander Taylor Jr. boarded at Cherbourg.
  Fact 0.385: Compton, Mr. Alexander Taylor Jr.'s destination is Lakewood, New Jersey, US.
  Fact 0.386: Cornell, Mrs. Malvina Helen (née Lamson) was 55 years old.
  Fact 0.387: Cornell, Mrs. Malvina Helen (née Lamson) is from New York City.
  Fact 0.388: Cornell, Mrs. Malvina Helen (née Lamson) boarded at Southampton.
  Fact 0.389: Cornell, Mrs. Malvina Helen (née Lamson)'s destination is New York City.
  Fact 0.390: Crafton, Mr. John Bertram was 59 years old.
  Fact 0.391: Crafton, Mr. John Bertram is from Roachdale, Indiana, US.
  Fact 0.392: Crafton, Mr. John Bertram boarded at Southampton.
  Fact 0.393: Crafton, Mr. John Bertram's destination is Roachdale, Indiana, US.
  Fact 0.394: Crosby, Captain Edward was 70 years old.
  Fact 0.395: Crosby, Captain Edward is from Milwaukee, Wisconsin, US.
  Fact 0.396: Crosby, Captain Edward boarded at Southampton.
  Fact 0.397: Crosby, Captain Edward's destination is Milwaukee, Wisconsin, US.
  Fact 0.398: Crosby, Captain Edward has 269 in the lifeboat or body column.
  Fact 0.399: Crosby, Mrs. Catherine Elizabeth (née Halstead) was 64 years old.
  Fact 0.400: Crosby, Mrs. Catherine Elizabeth (née Halstead) is from Milwaukee, Wisconsin, US.
  Fact 0.401: Crosby, Mrs. Catherine Elizabeth (née Halstead) boarded at Southampton.
  Fact 0.402: Crosby, Mrs. Catherine Elizabeth (née Halstead)'s destination is Milwaukee, Wisconsin, US.
  Fact 0.403: Crosby, Mrs. Catherine Elizabeth (née Halstead) has 7 in the lifeboat or body column.
  Fact 0.404: Crosby, Miss Harriette Rebecca was 39 years old.
  Fact 0.405: Crosby, Miss Harriette Rebecca is from Milwaukee, Wisconsin, US.
  Fact 0.406: Crosby, Miss Harriette Rebecca boarded at Southampton.
  Fact 0.407: Crosby, Miss Harriette Rebecca's destination is Milwaukee, Wisconsin, US.
  Fact 0.408: Crosby, Miss Harriette Rebecca has 7 in the lifeboat or body column.
  Fact 0.409: Cumings, Mr. John Bradley was 39 years old.
  Fact 0.410: Cumings, Mr. John Bradley is from New York City.
  Fact 0.411: Cumings, Mr. John Bradley boarded at Cherbourg.
  Fact 0.412: Cumings, Mr. John Bradley's destination is New York City.
  Fact 0.413: Cumings, Mrs. Florence Briggs (née Thayer) was 35 years old.
  Fact 0.414: Cumings, Mrs. Florence Briggs (née Thayer) is from New York City.
  Fact 0.415: Cumings, Mrs. Florence Briggs (née Thayer) boarded at Cherbourg.
  Fact 0.416: Cumings, Mrs. Florence Briggs (née Thayer)'s destination is New York, US.
  Fact 0.417: Daly, Mr. Peter Dennis was 51 years old.
  Fact 0.418: Daly, Mr. Peter Dennis is from Lima, Peru.
  Fact 0.419: Daly, Mr. Peter Dennis boarded at Southampton.
  Fact 0.420: Daly, Mr. Peter Dennis's destination is Lima, Peru.
  Fact 0.421: Daniel, Mr. Robert Williams was 27 years old.
  Fact 0.422: Daniel, Mr. Robert Williams is from Philadelphia, Pennsylvania, US.
  Fact 0.423: Daniel, Mr. Robert Williams boarded at Southampton.
  Fact 0.424: Daniel, Mr. Robert Williams's destination is Philadelphia, Pennsylvania, US.
  Fact 0.425: Davidson, Mr. Thornton was 31 years old.
  Fact 0.426: Davidson, Mr. Thornton is from Montreal, Quebec, Canada.
  Fact 0.427: Davidson, Mr. Thornton boarded at Southampton.
  Fact 0.428: Davidson, Mr. Thornton's destination is Montreal, Quebec, Canada.
  Fact 0.429: Davidson, Mrs. Orian (née Hays) was 27 years old.
  Fact 0.430: Davidson, Mrs. Orian (née Hays) is from Montreal, Quebec, Canada.
  Fact 0.431: Davidson, Mrs. Orian (née Hays) boarded at Southampton.
  Fact 0.432: Davidson, Mrs. Orian (née Hays)'s destination is Montreal, Quebec, Canada.
  Fact 0.433: Dick, Mr. Albert Adrian was 31 years old.
  Fact 0.434: Dick, Mr. Albert Adrian is from Calgary, Alberta, Canada.
  Fact 0.435: Dick, Mr. Albert Adrian boarded at Southampton.
  Fact 0.436: Dick, Mr. Albert Adrian's destination is Calgary, Alberta, Canada.
  Fact 0.437: Dick, Mrs. Vera (née Gillespie) was 17 years old.
  Fact 0.438: Dick, Mrs. Vera (née Gillespie) is from Calgary, Alberta, Canada.
  Fact 0.439: Dick, Mrs. Vera (née Gillespie) boarded at Southampton.
  Fact 0.440: Dick, Mrs. Vera (née Gillespie)'s destination is Calgary, Alberta, Canada.
  Fact 0.441: Dineston, Mrs. Nancy May Bowles was 28 years old.
  Fact 0.442: Dineston, Mrs. Nancy May Bowles is from London, England, UK.
  Fact 0.443: Dineston, Mrs. Nancy May Bowles boarded at Southampton.
  Fact 0.444: Dineston, Mrs. Nancy May Bowles's destination is New York City, USA.
  Fact 0.445: Dineston, George Samuel was 30 years old.
  Fact 0.446: Dineston, George Samuel is from London, England, UK.
  Fact 0.447: Dineston, George Samuel boarded at Southampton.
  Fact 0.448: Dineston, George Samuel's destination is New York City, USA.
  Fact 0.449: Dodge, Dr. Washington was 52 years old.
  Fact 0.450: Dodge, Dr. Washington is from San Francisco, California, US.
  Fact 0.451: Dodge, Dr. Washington boarded at Southampton.
  Fact 0.452: Dodge, Dr. Washington's destination is San Francisco, California, US.
  Fact 0.453: Dodge, Dr. Washington has 13 in the lifeboat or body column.
  Fact 0.454: Dodge, Mrs. Ruth (née Vidaver) was 34 years old.
  Fact 0.455: Dodge, Mrs. Ruth (née Vidaver) is from San Francisco, California, US.
  Fact 0.456: Dodge, Mrs. Ruth (née Vidaver) boarded at Southampton.
  Fact 0.457: Dodge, Mrs. Ruth (née Vidaver)'s destination is San Francisco, California, US.
  Fact 0.458: Dodge, Mrs. Ruth (née Vidaver) has 5 in the lifeboat or body column.
  Fact 0.459: Dodge, Master Washington, Jr. is from San Francisco, California, US.
  Fact 0.460: Dodge, Master Washington, Jr. boarded at Southampton.
  Fact 0.461: Dodge, Master Washington, Jr.'s destination is San Francisco, California, US.
  Fact 0.462: Dodge, Master Washington, Jr. has 5 in the lifeboat or body column.
  Fact 0.463: Douglas, Mr. Walter Donald was 50 years old.
  Fact 0.464: Douglas, Mr. Walter Donald is from Minneapolis, Minnesota, US.
  Fact 0.465: Douglas, Mr. Walter Donald boarded at Cherbourg.
  Fact 0.466: Douglas, Mr. Walter Donald's destination is Minneapolis, Minnesota, US.
  Fact 0.467: Douglas, Mr. Walter Donald has 62 in the lifeboat or body column.
  Fact 0.468: Douglas, Mrs. Mahala (née Dutton) was 48 years old.
  Fact 0.469: Douglas, Mrs. Mahala (née Dutton) is from Minneapolis, Minnesota, US.
  Fact 0.470: Douglas, Mrs. Mahala (née Dutton) boarded at Cherbourg.
  Fact 0.471: Douglas, Mrs. Mahala (née Dutton)'s destination is Minneapolis, Minnesota, US.
  Fact 0.472: Miss Berthe Leroy was 27 years old.
  Fact 0.473: Miss Berthe Leroy is listed as the maid.
  Fact 0.474: Miss Berthe Leroy is from Minneapolis, Minnesota, US.
  Fact 0.475: Miss Berthe Leroy boarded at Cherbourg.
  Fact 0.476: Miss Berthe Leroy's destination is Minneapolis, Minnesota, US.
  Fact 0.477: Douglas, Mrs. Mary Hélène (née Baxter) was 27 years old.
  Fact 0.478: Douglas, Mrs. Mary Hélène (née Baxter) is from Montreal, Quebec, Canada.
  Fact 0.479: Douglas, Mrs. Mary Hélène (née Baxter) boarded at Cherbourg.
  Fact 0.480: Douglas, Mrs. Mary Hélène (née Baxter)'s destination is Montreal, Quebec, Canada.
  Fact 0.481: Douglas, Mrs. Mary Hélène (née Baxter) has 6 in the lifeboat or body column.
  Fact 0.482: Duff Gordon, Sir Cosmo Edmund was 49 years old.
  Fact 0.483: Duff Gordon, Sir Cosmo Edmund is from London, England, UK.
  Fact 0.484: Duff Gordon, Sir Cosmo Edmund boarded at Cherbourg.
  Fact 0.485: Duff Gordon, Sir Cosmo Edmund's destination is New York City.
  Fact 0.486: Duff Gordon, Lucy Christiana, Lady (née Sutherland) was 48 years old.
  Fact 0.487: Duff Gordon, Lucy Christiana, Lady (née Sutherland) is from London, England, UK.
  Fact 0.488: Duff Gordon, Lucy Christiana, Lady (née Sutherland) boarded at Cherbourg.
  Fact 0.489: Duff Gordon, Lucy Christiana, Lady (née Sutherland)'s destination is New York City.
  Fact 0.490: Miss Laura Mabel Francatelli was 31 years old.
  Fact 0.491: Miss Laura Mabel Francatelli is listed as the secretary.
  Fact 0.492: Miss Laura Mabel Francatelli is from London, England, UK.
  Fact 0.493: Miss Laura Mabel Francatelli boarded at Cherbourg.
  Fact 0.494: Miss Laura Mabel Francatelli's destination is New York City.
  Fact 0.495: Dulles, Mr. William Crothers was 39 years old.
  Fact 0.496: Dulles, Mr. William Crothers is from Philadelphia, Pennsylvania, US.
  Fact 0.497: Dulles, Mr. William Crothers boarded at Cherbourg.
  Fact 0.498: Dulles, Mr. William Crothers's destination is Philadelphia, Pennsylvania, US.
  Fact 0.499: Dulles, Mr. William Crothers has 133 in the lifeboat or body column.
  Fact 0.500: Eustis, Miss Elizabeth Mussey was 54 years old.
  Fact 0.501: Eustis, Miss Elizabeth Mussey is from Brookline, Massachusetts, US.
  Fact 0.502: Eustis, Miss Elizabeth Mussey boarded at Cherbourg.
  Fact 0.503: Eustis, Miss Elizabeth Mussey's destination is Brookline, Massachusetts, US.
  Fact 0.504: Evans, Miss Edith Corse was 36 years old.
  Fact 0.505: Evans, Miss Edith Corse is from New York City.
  Fact 0.506: Evans, Miss Edith Corse boarded at Cherbourg.
  Fact 0.507: Evans, Miss Edith Corse's destination is New York City.
  Fact 0.508: Flegenheim, Mrs. Antoinette (née Wendt) was 48 years old.
  Fact 0.509: Flegenheim, Mrs. Antoinette (née Wendt) is from New York City.
  Fact 0.510: Flegenheim, Mrs. Antoinette (née Wendt) boarded at Cherbourg.
  Fact 0.511: Flegenheim, Mrs. Antoinette (née Wendt)'s destination is New York City.
  Fact 0.512: Flegenheim, Mrs. Antoinette (née Wendt) has 7 in the lifeboat or body column.
  Fact 0.513: Flynn, Mr. John Irwin was 36 years old.
  Fact 0.514: Flynn, Mr. John Irwin is from Brooklyn, New York, US.
  Fact 0.515: Flynn, Mr. John Irwin boarded at Southampton.
  Fact 0.516: Flynn, Mr. John Irwin's destination is Brooklyn, New York, US.
  Fact 0.517: Flynn, Mr. John Irwin has 5 in the lifeboat or body column.
  Fact 0.518: Foreman, Mr. Benjamin Laventall was 30 years old.
  Fact 0.519: Foreman, Mr. Benjamin Laventall is from New York City.
  Fact 0.520: Foreman, Mr. Benjamin Laventall boarded at Southampton.
  Fact 0.521: Foreman, Mr. Benjamin Laventall's destination is New York City.
  Fact 0.522: Fortune, Mr. Mark was 64 years old.
  Fact 0.523: Fortune, Mr. Mark is from Winnipeg, Manitoba, Canada.
  Fact 0.524: Fortune, Mr. Mark boarded at Southampton.
  Fact 0.525: Fortune, Mr. Mark's destination is Winnipeg, Manitoba, Canada.
  Fact 0.526: Fortune, Mrs. Mary (née McDougald) was 60 years old.
  Fact 0.527: Fortune, Mrs. Mary (née McDougald) is from Winnipeg, Manitoba, Canada.
  Fact 0.528: Fortune, Mrs. Mary (née McDougald) boarded at Southampton.
  Fact 0.529: Fortune, Mrs. Mary (née McDougald)'s destination is Winnipeg, Manitoba, Canada.
  Fact 0.530: Fortune, Mrs. Mary (née McDougald) has 10 in the lifeboat or body column.
  Fact 0.531: Fortune, Miss Ethel Flora was 28 years old.
  Fact 0.532: Fortune, Miss Ethel Flora is from Winnipeg, Manitoba, Canada.
  Fact 0.533: Fortune, Miss Ethel Flora boarded at Southampton.
  Fact 0.534: Fortune, Miss Ethel Flora's destination is Winnipeg, Manitoba, Canada.
  Fact 0.535: Fortune, Miss Ethel Flora has 10 in the lifeboat or body column.
  Fact 0.536: Fortune, Miss Alice Elizabeth was 24 years old.
  Fact 0.537: Fortune, Miss Alice Elizabeth is from Winnipeg, Manitoba, Canada.
  Fact 0.538: Fortune, Miss Alice Elizabeth boarded at Southampton.
  Fact 0.539: Fortune, Miss Alice Elizabeth's destination is Winnipeg, Manitoba, Canada.
  Fact 0.540: Fortune, Miss Alice Elizabeth has 10 in the lifeboat or body column.
  Fact 0.541: Fortune, Miss Mabel Helen was 23 years old.
  Fact 0.542: Fortune, Miss Mabel Helen is from Winnipeg, Manitoba, Canada.
  Fact 0.543: Fortune, Miss Mabel Helen boarded at Southampton.
  Fact 0.544: Fortune, Miss Mabel Helen's destination is Winnipeg, Manitoba, Canada.
  Fact 0.545: Fortune, Miss Mabel Helen has 10 in the lifeboat or body column.
  Fact 0.546: Fortune, Mr. Charles Alexander was 19 years old.
  Fact 0.547: Fortune, Mr. Charles Alexander is from Winnipeg, Manitoba, Canada.
  Fact 0.548: Fortune, Mr. Charles Alexander boarded at Southampton.
  Fact 0.549: Fortune, Mr. Charles Alexander's destination is Winnipeg, Manitoba, Canada.
  Fact 0.550: Franklin, Mr. Thomas Parnham was 37 years old.
  Fact 0.551: Franklin, Mr. Thomas Parnham is from London, England, UK.
  Fact 0.552: Franklin, Mr. Thomas Parnham boarded at Southampton.
  Fact 0.553: Franklin, Mr. Thomas Parnham's destination is New York City.
  Fact 0.554: Frauenthal, Dr. Henry William was 49 years old.
  Fact 0.555: Frauenthal, Dr. Henry William is from London, England, UK.
  Fact 0.556: Frauenthal, Dr. Henry William boarded at Southampton.
  Fact 0.557: Frauenthal, Dr. Henry William's destination is New York City.
  Fact 0.558: Frauenthal, Dr. Henry William has 5 in the lifeboat or body column.
  Fact 0.559: Frauenthal, Mrs. Clara (née Heinsheimer) was 42 years old.
  Fact 0.560: Frauenthal, Mrs. Clara (née Heinsheimer) is from London, England, UK.
  Fact 0.561: Frauenthal, Mrs. Clara (née Heinsheimer) boarded at Southampton.
  Fact 0.562: Frauenthal, Mrs. Clara (née Heinsheimer)'s destination is New York City.
  Fact 0.563: Frauenthal, Mrs. Clara (née Heinsheimer) has 5 in the lifeboat or body column.
  Fact 0.564: Frauenthal, Mr. Isaac Gerald was 43 years old.
  Fact 0.565: Frauenthal, Mr. Isaac Gerald is from London, England, UK.
  Fact 0.566: Frauenthal, Mr. Isaac Gerald boarded at Southampton.
  Fact 0.567: Frauenthal, Mr. Isaac Gerald's destination is New York City.
  Fact 0.568: Frauenthal, Mr. Isaac Gerald has 5 in the lifeboat or body column.
  Fact 0.569: Frölicher, Mr. Maximilian Josef was 60 years old.
  Fact 0.570: Frölicher, Mr. Maximilian Josef is from Zürich, Switzerland.
  Fact 0.571: Frölicher, Mr. Maximilian Josef boarded at Cherbourg.
  Fact 0.572: Frölicher, Mr. Maximilian Josef's destination is New York City.
  Fact 0.573: Frölicher, Mr. Maximilian Josef has 5 in the lifeboat or body column.
  Fact 0.574: Frölicher, Mrs. Margaretha Emerentia (née Stehli) was 48 years old.
  Fact 0.575: Frölicher, Mrs. Margaretha Emerentia (née Stehli) is from Zürich, Switzerland.
  Fact 0.576: Frölicher, Mrs. Margaretha Emerentia (née Stehli) boarded at Cherbourg.
  Fact 0.577: Frölicher, Mrs. Margaretha Emerentia (née Stehli)'s destination is New York City.
  Fact 0.578: Frölicher, Mrs. Margaretha Emerentia (née Stehli) has 5 in the lifeboat or body column.
  Fact 0.579: Frölicher-Stehli, Miss Hedwig Margaritha was 22 years old.
  Fact 0.580: Frölicher-Stehli, Miss Hedwig Margaritha is from Zürich, Switzerland.
  Fact 0.581: Frölicher-Stehli, Miss Hedwig Margaritha boarded at Cherbourg.
  Fact 0.582: Frölicher-Stehli, Miss Hedwig Margaritha's destination is New York City.
  Fact 0.583: Frölicher-Stehli, Miss Hedwig Margaritha has 5 in the lifeboat or body column.
  Fact 0.584: Futrelle, Mr. Jacques Heath was 37 years old.
  Fact 0.585: Futrelle, Mr. Jacques Heath is from Scituate, Massachusetts, US.
  Fact 0.586: Futrelle, Mr. Jacques Heath boarded at Southampton.
  Fact 0.587: Futrelle, Mr. Jacques Heath's destination is Scituate, Massachusetts, US.
  Fact 0.588: Futrelle, Mrs. Lily May (née Peel) was 35 years old.
  Fact 0.589: Futrelle, Mrs. Lily May (née Peel) is from Scituate, Massachusetts, US.
  Fact 0.590: Futrelle, Mrs. Lily May (née Peel) boarded at Southampton.
  Fact 0.591: Futrelle, Mrs. Lily May (née Peel)'s destination is Scituate, Massachusetts, US.
  Fact 0.592: Gee, Mr. Arthur H. was 47 years old.
  Fact 0.593: Gee, Mr. Arthur H. is from St. Annes-on-Sea, Lancashire, England, UK.
  Fact 0.594: Gee, Mr. Arthur H. boarded at Southampton.
  Fact 0.595: Gee, Mr. Arthur H.'s destination is Mexico City, Mexico.
  Fact 0.596: Gee, Mr. Arthur H. has 275 in the lifeboat or body column.
  Fact 0.597: Gibson, Mrs. Pauline Caroline (née Boeson) was 44 years old.
  Fact 0.598: Gibson, Mrs. Pauline Caroline (née Boeson) is from New York City.
  Fact 0.599: Gibson, Mrs. Pauline Caroline (née Boeson) boarded at Cherbourg.
  Fact 0.600: Gibson, Mrs. Pauline Caroline (née Boeson)'s destination is New York City.
  Fact 0.601: Gibson, Mrs. Pauline Caroline (née Boeson) has 7 in the lifeboat or body column.
  Fact 0.602: Gibson, Miss Dorothy Winifred was 22 years old.
  Fact 0.603: Gibson, Miss Dorothy Winifred is from New York City.
  Fact 0.604: Gibson, Miss Dorothy Winifred boarded at Cherbourg.
  Fact 0.605: Gibson, Miss Dorothy Winifred's destination is New York City.
  Fact 0.606: Gibson, Miss Dorothy Winifred has 7 in the lifeboat or body column.
  Fact 0.607: Goldenberg, Mr. Samuel L. was 47 years old.
  Fact 0.608: Goldenberg, Mr. Samuel L. is from Paris, France.
  Fact 0.609: Goldenberg, Mr. Samuel L. boarded at Cherbourg.
  Fact 0.610: Goldenberg, Mr. Samuel L.'s destination is New York City.
  Fact 0.611: Goldenberg, Mr. Samuel L. has 5 in the lifeboat or body column.
  Fact 0.612: Goldenberg, Mrs. Nella (née Wiggins) was 40 years old.
  Fact 0.613: Goldenberg, Mrs. Nella (née Wiggins) is from Paris, France.
  Fact 0.614: Goldenberg, Mrs. Nella (née Wiggins) boarded at Cherbourg.
  Fact 0.615: Goldenberg, Mrs. Nella (née Wiggins)'s destination is New York City.
  Fact 0.616: Goldenberg, Mrs. Nella (née Wiggins) has 5 in the lifeboat or body column.
  Fact 0.617: Goldschmidt, Mr. George B. was 71 years old.
  Fact 0.618: Goldschmidt, Mr. George B. is from New York City.
  Fact 0.619: Goldschmidt, Mr. George B. boarded at Cherbourg.
  Fact 0.620: Goldschmidt, Mr. George B.'s destination is New York City.
  Fact 0.621: Gracie IV, Colonel Archibald was 53 years old.
  Fact 0.622: Gracie IV, Colonel Archibald is from Washington, D.C., US.
  Fact 0.623: Gracie IV, Colonel Archibald boarded at Southampton.
  Fact 0.624: Gracie IV, Colonel Archibald's destination is Washington, D.C., US.
  Fact 0.625: Graham, Mr. George Edward was 38 years old.
  Fact 0.626: Graham, Mr. George Edward is from Winnipeg, Manitoba, Canada.
  Fact 0.627: Graham, Mr. George Edward boarded at Southampton.
  Fact 0.628: Graham, Mr. George Edward's destination is Winnipeg, Manitoba, Canada.
  Fact 0.629: Graham, Mr. George Edward has 147 in the lifeboat or body column.
  Fact 0.630: Graham, Mrs. Edith Ware (née Junkins) was 59 years old.
  Fact 0.631: Graham, Mrs. Edith Ware (née Junkins) is from Greenwich, Connecticut, US.
  Fact 0.632: Graham, Mrs. Edith Ware (née Junkins) boarded at Southampton.
  Fact 0.633: Graham, Mrs. Edith Ware (née Junkins)'s destination is Greenwich, Connecticut, US.
  Fact 0.634: Graham, Miss Margaret Edith was 19 years old.
  Fact 0.635: Graham, Miss Margaret Edith is from Greenwich, Connecticut, US.
  Fact 0.636: Graham, Miss Margaret Edith boarded at Southampton.
  Fact 0.637: Graham, Miss Margaret Edith's destination is Greenwich, Connecticut, US.
  Fact 0.638: Miss Elizabeth Weed Shutes was 40 years old.
  Fact 0.639: Miss Elizabeth Weed Shutes is listed as the governess.
  Fact 0.640: Miss Elizabeth Weed Shutes is from Greenwich, Connecticut, US.
  Fact 0.641: Miss Elizabeth Weed Shutes boarded at Southampton.
  Fact 0.642: Miss Elizabeth Weed Shutes's destination is Greenwich, Connecticut, US.
  Fact 0.643: Greenfield, Mrs. Blanche (née Strouse) was 45 years old.
  Fact 0.644: Greenfield, Mrs. Blanche (née Strouse) is from New York City.
  Fact 0.645: Greenfield, Mrs. Blanche (née Strouse) boarded at Cherbourg.
  Fact 0.646: Greenfield, Mrs. Blanche (née Strouse)'s destination is New York City.
  Fact 0.647: Greenfield, Mrs. Blanche (née Strouse) has 7 in the lifeboat or body column.
  Fact 0.648: Greenfield, Mr. William Bertram was 23 years old.
  Fact 0.649: Greenfield, Mr. William Bertram is from New York City.
  Fact 0.650: Greenfield, Mr. William Bertram boarded at Cherbourg.
  Fact 0.651: Greenfield, Mr. William Bertram's destination is New York City.
  Fact 0.652: Greenfield, Mr. William Bertram has 7 in the lifeboat or body column.
  Fact 0.653: Guggenheim, Mr. Benjamin was 46 years old.
  Fact 0.654: Guggenheim, Mr. Benjamin is from Paris, France.
  Fact 0.655: Guggenheim, Mr. Benjamin boarded at Cherbourg.
  Fact 0.656: Guggenheim, Mr. Benjamin's destination is New York City.
  Fact 0.657: Mr. Victor Giglio was 24 years old.
  Fact 0.658: Mr. Victor Giglio is listed as the valet.
  Fact 0.659: Mr. Victor Giglio is from Paris, France.
  Fact 0.660: Mr. Victor Giglio boarded at Cherbourg.
  Fact 0.661: Mr. Victor Giglio's destination is New York City.
  Fact 0.662: Mr. René Pernot was 39 years old.
  Fact 0.663: Mr. René Pernot is listed as the chauffeur.
  Fact 0.664: Mr. René Pernot is from Paris, France.
  Fact 0.665: Mr. René Pernot boarded at Cherbourg.
  Fact 0.666: Mr. René Pernot's destination is New York City.
  Fact 0.667: Harder, Mr. George Achilles was 25 years old.
  Fact 0.668: Harder, Mr. George Achilles is from New York City.
  Fact 0.669: Harder, Mr. George Achilles boarded at Cherbourg.
  Fact 0.670: Harder, Mr. George Achilles's destination is New York City.
  Fact 0.671: Harder, Mr. George Achilles has 5 in the lifeboat or body column.
  Fact 0.672: Harder, Mrs. Dorothy (née Annan) was 21 years old.
  Fact 0.673: Harder, Mrs. Dorothy (née Annan) is from New York City.
  Fact 0.674: Harder, Mrs. Dorothy (née Annan) boarded at Cherbourg.
  Fact 0.675: Harder, Mrs. Dorothy (née Annan)'s destination is New York City.
  Fact 0.676: Harder, Mrs. Dorothy (née Annan) has 5 in the lifeboat or body column.
  Fact 0.677: Harper, Mr. Henry Sleeper was 48 years old.
  Fact 0.678: Harper, Mr. Henry Sleeper is from New York City.
  Fact 0.679: Harper, Mr. Henry Sleeper boarded at Cherbourg.
  Fact 0.680: Harper, Mr. Henry Sleeper's destination is New York City.
  Fact 0.681: Mr. Hammad Hassab was 27 years old.
  Fact 0.682: Mr. Hammad Hassab is listed as the dragoman.
  Fact 0.683: Mr. Hammad Hassab is from Cairo, Egypt.
  Fact 0.684: Mr. Hammad Hassab boarded at Cherbourg.
  Fact 0.685: Mr. Hammad Hassab's destination is New York City.
  Fact 0.686: Harper, Mrs. Myna (née Haxtun) was 49 years old.
  Fact 0.687: Harper, Mrs. Myna (née Haxtun) is from New York City.
  Fact 0.688: Harper, Mrs. Myna (née Haxtun) boarded at Cherbourg.
  Fact 0.689: Harper, Mrs. Myna (née Haxtun)'s destination is New York City.
  Fact 0.690: Harris, Mr. Henry Birkhardt was 45 years old.
  Fact 0.691: Harris, Mr. Henry Birkhardt is from New York City.
  Fact 0.692: Harris, Mr. Henry Birkhardt boarded at Southampton.
  Fact 0.693: Harris, Mr. Henry Birkhardt's destination is New York City.
  Fact 0.694: Harris, Mrs. Irene (née Wallach) was 35 years old.
  Fact 0.695: Harris, Mrs. Irene (née Wallach) is from New York City.
  Fact 0.696: Harris, Mrs. Irene (née Wallach) boarded at Southampton.
  Fact 0.697: Harris, Mrs. Irene (née Wallach)'s destination is New York City.
  Fact 0.698: Hawksford, Mr. Walter James was 45 years old.
  Fact 0.699: Hawksford, Mr. Walter James is from Kingston, Surrey, England, UK.
  Fact 0.700: Hawksford, Mr. Walter James boarded at Southampton.
  Fact 0.701: Hawksford, Mr. Walter James's destination is New York City.
  Fact 0.702: Hays, Mr. Charles Melville was 55 years old.
  Fact 0.703: Hays, Mr. Charles Melville is from Montreal, Quebec, Canada.
  Fact 0.704: Hays, Mr. Charles Melville boarded at Southampton.
  Fact 0.705: Hays, Mr. Charles Melville's destination is Montreal, Quebec, Canada.
  Fact 0.706: Hays, Mr. Charles Melville has 307 in the lifeboat or body column.
  Fact 0.707: Mr. Vivian Ponsonby Payne was 22 years old.
  Fact 0.708: Mr. Vivian Ponsonby Payne is listed as the clerk.
  Fact 0.709: Mr. Vivian Ponsonby Payne is from Montreal, Quebec, Canada.
  Fact 0.710: Mr. Vivian Ponsonby Payne boarded at Southampton.
  Fact 0.711: Mr. Vivian Ponsonby Payne's destination is Montreal, Quebec, Canada.
  Fact 0.712: Hays, Mrs. Clara Jennings (née Grigg) was 52 years old.
  Fact 0.713: Hays, Mrs. Clara Jennings (née Grigg) is from Montreal, Quebec, Canada.
  Fact 0.714: Hays, Mrs. Clara Jennings (née Grigg) boarded at Southampton.
  Fact 0.715: Hays, Mrs. Clara Jennings (née Grigg)'s destination is Montreal, Quebec, Canada.
  Fact 0.716: Miss Mary Anne Perreault was 33 years old.
  Fact 0.717: Miss Mary Anne Perreault is listed as the maid.
  Fact 0.718: Miss Mary Anne Perreault is from Montreal, Quebec, Canada.
  Fact 0.719: Miss Mary Anne Perreault boarded at Southampton.
  Fact 0.720: Miss Mary Anne Perreault's destination is Montreal, Quebec, Canada.
  Fact 0.721: Hays, Miss Margaret Bechstein was 24 years old.
  Fact 0.722: Hays, Miss Margaret Bechstein is from New York City.
  Fact 0.723: Hays, Miss Margaret Bechstein boarded at Cherbourg.
  Fact 0.724: Hays, Miss Margaret Bechstein's destination is New York City.
  Fact 0.725: Hays, Miss Margaret Bechstein has 7 in the lifeboat or body column.
  Fact 0.726: Head, Mr. Christopher was 42 years old.
  Fact 0.727: Head, Mr. Christopher is from Chelsea, London, England, UK.
  Fact 0.728: Head, Mr. Christopher boarded at Southampton.
  Fact 0.729: Head, Mr. Christopher's destination is New York City.
  Fact 0.730: Hilliard, Mr. Herbert Henry was 44 years old.
  Fact 0.731: Hilliard, Mr. Herbert Henry is from Brighton, Massachusetts, US.
  Fact 0.732: Hilliard, Mr. Herbert Henry boarded at Southampton.
  Fact 0.733: Hilliard, Mr. Herbert Henry's destination is Brighton, Massachusetts, US.
  Fact 0.734: Hipkins, Mr. William Edward was 55 years old.
  Fact 0.735: Hipkins, Mr. William Edward is from Birmingham, West Midlands, England, UK.
  Fact 0.736: Hipkins, Mr. William Edward boarded at Southampton.
  Fact 0.737: Hipkins, Mr. William Edward's destination is New York City.
  Fact 0.738: Hippach, Mrs. Ida Sophia (née Fischer) was 44 years old.
  Fact 0.739: Hippach, Mrs. Ida Sophia (née Fischer) is from Chicago, Illinois, US.
  Fact 0.740: Hippach, Mrs. Ida Sophia (née Fischer) boarded at Cherbourg.
  Fact 0.741: Hippach, Mrs. Ida Sophia (née Fischer)'s destination is Chicago, Illinois, US.
  Fact 0.742: Hippach, Miss Jean Gertrude was 17 years old.
  Fact 0.743: Hippach, Miss Jean Gertrude is from Chicago, Illinois, US.
  Fact 0.744: Hippach, Miss Jean Gertrude boarded at Cherbourg.
  Fact 0.745: Hippach, Miss Jean Gertrude's destination is Chicago, Illinois, US.
  Fact 0.746: Hogeboom, Mrs. Anna Louisa (née Andrews) was 51 years old.
  Fact 0.747: Hogeboom, Mrs. Anna Louisa (née Andrews) is from Hudson, New York, US.
  Fact 0.748: Hogeboom, Mrs. Anna Louisa (née Andrews) boarded at Cherbourg.
  Fact 0.749: Hogeboom, Mrs. Anna Louisa (née Andrews)'s destination is Hudson, New York, US.
  Fact 0.750: Hogeboom, Mrs. Anna Louisa (née Andrews) has 10 in the lifeboat or body column.
  Fact 0.751: Holverson, Mr. Alexander Oskar was 42 years old.
  Fact 0.752: Holverson, Mr. Alexander Oskar is from New York City.
  Fact 0.753: Holverson, Mr. Alexander Oskar boarded at Southampton.
  Fact 0.754: Holverson, Mr. Alexander Oskar's destination is New York City.
  Fact 0.755: Holverson, Mr. Alexander Oskar has 38 in the lifeboat or body column.
  Fact 0.756: Holverson, Mrs. Mary Aline (née Towner) was 35 years old.
  Fact 0.757: Holverson, Mrs. Mary Aline (née Towner) is from New York City.
  Fact 0.758: Holverson, Mrs. Mary Aline (née Towner) boarded at Southampton.
  Fact 0.759: Holverson, Mrs. Mary Aline (née Towner)'s destination is New York City.
  Fact 0.760: Holverson, Mrs. Mary Aline (née Towner) has 8 in the lifeboat or body column.
  Fact 0.761: Homer, Mr. Harry (alias E. Haven) was 40 years old.
  Fact 0.762: Homer, Mr. Harry (alias E. Haven) is from Indianapolis, Indiana, US.
  Fact 0.763: Homer, Mr. Harry (alias E. Haven) boarded at Southampton.
  Fact 0.764: Homer, Mr. Harry (alias E. Haven)'s destination is Indianapolis, Indiana, US.
  Fact 0.765: Homer, Mr. Harry (alias E. Haven) has 15 in the lifeboat or body column.
  Fact 0.766: Hoyt, Mr. Frederick Maxfield was 35 years old.
  Fact 0.767: Hoyt, Mr. Frederick Maxfield is from New York City.
  Fact 0.768: Hoyt, Mr. Frederick Maxfield boarded at Southampton.
  Fact 0.769: Hoyt, Mr. Frederick Maxfield's destination is Stamford, Connecticut, US.
  Fact 0.770: Hoyt, Mrs. Jane Anne (née Forby) was 31 years old.
  Fact 0.771: Hoyt, Mrs. Jane Anne (née Forby) is from New York City.
  Fact 0.772: Hoyt, Mrs. Jane Anne (née Forby) boarded at Southampton.
  Fact 0.773: Hoyt, Mrs. Jane Anne (née Forby)'s destination is Stamford, Connecticut, US.
  Fact 0.774: Hoyt, Mr. William Fisher was 42 years old.
  Fact 0.775: Hoyt, Mr. William Fisher is from New York City.
  Fact 0.776: Hoyt, Mr. William Fisher boarded at Cherbourg.
  Fact 0.777: Hoyt, Mr. William Fisher's destination is New York City.
  Fact 0.778: Hoyt, Mr. William Fisher has 14 in the lifeboat or body column.
  Fact 0.779: Isham, Miss Ann Elizabeth was 50 years old.
  Fact 0.780: Isham, Miss Ann Elizabeth is from Chicago, Illinois, US.
  Fact 0.781: Isham, Miss Ann Elizabeth boarded at Cherbourg.
  Fact 0.782: Isham, Miss Ann Elizabeth's destination is Chicago, Illinois, US.
  Fact 0.783: Ismay, Mr. Joseph Bruce was 49 years old.
  Fact 0.784: Ismay, Mr. Joseph Bruce is from Liverpool, Merseyside, England, UK.
  Fact 0.785: Ismay, Mr. Joseph Bruce boarded at Southampton.
  Fact 0.786: Ismay, Mr. Joseph Bruce's destination is New York City.
  Fact 0.787: Mr. John Richard Fry was 39 years old.
  Fact 0.788: Mr. John Richard Fry is listed as the valet.
  Fact 0.789: Mr. John Richard Fry is from Liverpool, Merseyside, England, UK.
  Fact 0.790: Mr. John Richard Fry boarded at Southampton.
  Fact 0.791: Mr. John Richard Fry's destination is New York City.
  Fact 0.792: Mr. William Henry Harrison was 45 years old.
  Fact 0.793: Mr. William Henry Harrison is listed as the secretary.
  Fact 0.794: Mr. William Henry Harrison is from Wallasey, Merseyside, England, UK.
  Fact 0.795: Mr. William Henry Harrison boarded at Southampton.
  Fact 0.796: Mr. William Henry Harrison's destination is New York City.
  Fact 0.797: Mr. William Henry Harrison has 110 in the lifeboat or body column.
  Fact 0.798: Johnson, Mrs. Madeline Eliza (née Austin) was 40 years old.
  Fact 0.799: Johnson, Mrs. Madeline Eliza (née Austin) is from London, England, UK.
  Fact 0.800: Johnson, Mrs. Madeline Eliza (née Austin) boarded at Southampton.
  Fact 0.801: Johnson, Mrs. Madeline Eliza (née Austin)'s destination is New York, New York, US.
  Fact 0.802: Johnson, Mrs. Madeline Eliza (née Austin) has 6 in the lifeboat or body column.
  Fact 0.803: Johnson (later changed to Langston), Mr. Samuel James was 10 years old.
  Fact 0.804: Johnson (later changed to Langston), Mr. Samuel James is from London, England, UK.
  Fact 0.805: Johnson (later changed to Langston), Mr. Samuel James boarded at Southampton.
  Fact 0.806: Johnson (later changed to Langston), Mr. Samuel James's destination is New York, New York, US.
  Fact 0.807: Jones, Mr. Charles Cresson was 46 years old.
  Fact 0.808: Jones, Mr. Charles Cresson is from Bennington, Vermont, US.
  Fact 0.809: Jones, Mr. Charles Cresson boarded at Southampton.
  Fact 0.810: Jones, Mr. Charles Cresson's destination is Bennington, Vermont, US.
  Fact 0.811: Jones, Mr. Charles Cresson has 80 in the lifeboat or body column.
  Fact 0.812: Julian, Mr. Henry Forbes was 50 years old.
  Fact 0.813: Julian, Mr. Henry Forbes is from Torquay, Devon, England, UK.
  Fact 0.814: Julian, Mr. Henry Forbes boarded at Southampton.
  Fact 0.815: Julian, Mr. Henry Forbes's destination is San Francisco, US.
  Fact 0.816: Kent, Mr. Edward Austin was 58 years old.
  Fact 0.817: Kent, Mr. Edward Austin is from Buffalo, New York, US.
  Fact 0.818: Kent, Mr. Edward Austin boarded at Cherbourg.
  Fact 0.819: Kent, Mr. Edward Austin's destination is Buffalo, New York, US.
  Fact 0.820: Kent, Mr. Edward Austin has 258 in the lifeboat or body column.
  Fact 0.821: Kenyon, Mr. Frederick R. was 41 years old.
  Fact 0.822: Kenyon, Mr. Frederick R. is from Pittsburgh, Pennsylvania, US.
  Fact 0.823: Kenyon, Mr. Frederick R. boarded at Southampton.
  Fact 0.824: Kenyon, Mr. Frederick R.'s destination is Pittsburgh, Pennsylvania, US.
  Fact 0.825: Kenyon, Mrs. Marion (née Stauffer) was 31 years old.
  Fact 0.826: Kenyon, Mrs. Marion (née Stauffer) is from Pittsburgh, Pennsylvania, US.
  Fact 0.827: Kenyon, Mrs. Marion (née Stauffer) boarded at Southampton.
  Fact 0.828: Kenyon, Mrs. Marion (née Stauffer)'s destination is Pittsburgh, Pennsylvania, US.
  Fact 0.829: Kenyon, Mrs. Marion (née Stauffer) has 8 in the lifeboat or body column.
  Fact 0.830: Kimball, Mr. Edwin Nelson Jr. was 42 years old.
  Fact 0.831: Kimball, Mr. Edwin Nelson Jr. is from Boston, Massachusetts, US.
  Fact 0.832: Kimball, Mr. Edwin Nelson Jr. boarded at Southampton.
  Fact 0.833: Kimball, Mr. Edwin Nelson Jr.'s destination is Boston, Massachusetts, US.
  Fact 0.834: Kimball, Mr. Edwin Nelson Jr. has 5 in the lifeboat or body column.
  Fact 0.835: Kimball, Mrs. Gertrude (née Parsons) was 45 years old.
  Fact 0.836: Kimball, Mrs. Gertrude (née Parsons) is from Boston, Massachusetts, US.
  Fact 0.837: Kimball, Mrs. Gertrude (née Parsons) boarded at Southampton.
  Fact 0.838: Kimball, Mrs. Gertrude (née Parsons)'s destination is Boston, Massachusetts, US.
  Fact 0.839: Kimball, Mrs. Gertrude (née Parsons) has 5 in the lifeboat or body column.
  Fact 0.840: Klaber, Mr. Herman was 45 years old.
  Fact 0.841: Klaber, Mr. Herman is from Portland, Oregon, US.
  Fact 0.842: Klaber, Mr. Herman boarded at Southampton.
  Fact 0.843: Klaber, Mr. Herman's destination is Portland, Oregon, US.
  Fact 0.844: Langston, Ms. Maxine Jane was 25 years old.
  Fact 0.845: Langston, Ms. Maxine Jane is from Duncansville, Pennsylvania, US.
  Fact 0.846: Langston, Ms. Maxine Jane boarded at Southampton.
  Fact 0.847: Langston, Ms. Maxine Jane's destination is Duncansivlle, Pennsylvania, US.
  Fact 0.848: Lambert-Williams, Mr. Fletcher Fellows was 43 years old.
  Fact 0.849: Lambert-Williams, Mr. Fletcher Fellows is from London, England, UK.
  Fact 0.850: Lambert-Williams, Mr. Fletcher Fellows boarded at Southampton.
  Fact 0.851: Lambert-Williams, Mr. Fletcher Fellows's destination is Newark, New Jersey, US.
  Fact 0.852: Leader, Dr. Alice (née Farnham) was 49 years old.
  Fact 0.853: Leader, Dr. Alice (née Farnham) is from New York City.
  Fact 0.854: Leader, Dr. Alice (née Farnham) boarded at Southampton.
  Fact 0.855: Leader, Dr. Alice (née Farnham)'s destination is New York City.
  Fact 0.856: Leader, Dr. Alice (née Farnham) has 8 in the lifeboat or body column.
  Fact 0.857: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) was 27 years old.
  Fact 0.858: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) is from Vancouver, British Columbia, Canada.
  Fact 0.859: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) boarded at Southampton.
  Fact 0.860: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes)'s destination is Vancouver, British Columbia, Canada.
  Fact 0.861: Leslie, Lucy Noël Martha, Countess of Rothes (née Dyer-Edwardes) has 8 in the lifeboat or body column.
  Fact 0.862: Miss Roberta Elizabeth Mary "Cissy" Maioni was 20 years old.
  Fact 0.863: Miss Roberta Elizabeth Mary "Cissy" Maioni is listed as the maid.
  Fact 0.864: Miss Roberta Elizabeth Mary "Cissy" Maioni is from Vancouver, British Columbia, Canada.
  Fact 0.865: Miss Roberta Elizabeth Mary "Cissy" Maioni boarded at Southampton.
  Fact 0.866: Miss Roberta Elizabeth Mary "Cissy" Maioni's destination is Vancouver, British Columbia, Canada.
  Fact 0.867: Miss Roberta Elizabeth Mary "Cissy" Maioni has 8 in the lifeboat or body column.
  Fact 0.868: Lewy, Mr. Ervin G. was 31 years old.
  Fact 0.869: Lewy, Mr. Ervin G. is from Chicago, Illinois, US.
  Fact 0.870: Lewy, Mr. Ervin G. boarded at Cherbourg.
  Fact 0.871: Lewy, Mr. Ervin G.'s destination is Chicago, Illinois, US.
  Fact 0.872: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) was 42 years old.
  Fact 0.873: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) is from Jordanstorp, Södermanland, Sweden.
  Fact 0.874: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey) boarded at Southampton.
  Fact 0.875: Lindeberg-Lind, Mr. Erik Gustav (alias Edward Lingrey)'s destination is New York City.
  Fact 0.876: Lindström, Mrs. Sigrid (née Posse) was 55 years old.
  Fact 0.877: Lindström, Mrs. Sigrid (née Posse) is from Stockholm, Sweden.
  Fact 0.878: Lindström, Mrs. Sigrid (née Posse) boarded at Cherbourg.
  Fact 0.879: Lindström, Mrs. Sigrid (née Posse)'s destination is New York City.
  Fact 0.880: Lindström, Mrs. Sigrid (née Posse) has 6 in the lifeboat or body column.
  Fact 0.881: Lines, Mrs. Elizabeth Lindsey (née James) was 50 years old.
  Fact 0.882: Lines, Mrs. Elizabeth Lindsey (née James) is from Paris, France.
  Fact 0.883: Lines, Mrs. Elizabeth Lindsey (née James) boarded at Cherbourg.
  Fact 0.884: Lines, Mrs. Elizabeth Lindsey (née James)'s destination is Hanover, New Hampshire, US.
  Fact 0.885: Lines, Mrs. Elizabeth Lindsey (née James) has 9 in the lifeboat or body column.
  Fact 0.886: Lines, Miss Mary Conover was 16 years old.
  Fact 0.887: Lines, Miss Mary Conover is from Paris, France.
  Fact 0.888: Lines, Miss Mary Conover boarded at Cherbourg.
  Fact 0.889: Lines, Miss Mary Conover's destination is Hanover, New Hampshire, US.
  Fact 0.890: Lines, Miss Mary Conover has 9 in the lifeboat or body column.
  Fact 0.891: Long, Mr. Milton Clyde was 29 years old.
  Fact 0.892: Long, Mr. Milton Clyde is from Springfield, Massachusetts, US.
  Fact 0.893: Long, Mr. Milton Clyde boarded at Southampton.
  Fact 0.894: Long, Mr. Milton Clyde's destination is Springfield, Massachusetts, US.
  Fact 0.895: Long, Mr. Milton Clyde has 126 in the lifeboat or body column.
  Fact 0.896: Longley, Miss Gretchen Fiske was 21 years old.
  Fact 0.897: Longley, Miss Gretchen Fiske is from Hudson, New York, US.
  Fact 0.898: Longley, Miss Gretchen Fiske boarded at Cherbourg.
  Fact 0.899: Longley, Miss Gretchen Fiske's destination is Hudson, New York, US.
  Fact 0.900: Longley, Miss Gretchen Fiske has 10 in the lifeboat or body column.
  Fact 0.901: Loring, Mr. Joseph Holland was 30 years old.
  Fact 0.902: Loring, Mr. Joseph Holland is from London, England, UK.
  Fact 0.903: Loring, Mr. Joseph Holland boarded at Southampton.
  Fact 0.904: Loring, Mr. Joseph Holland's destination is New York City.
  Fact 0.905: Madill, Miss Georgette Alexandra was 16 years old.
  Fact 0.906: Madill, Miss Georgette Alexandra is from St. Louis, Missouri, US.
  Fact 0.907: Madill, Miss Georgette Alexandra boarded at Southampton.
  Fact 0.908: Madill, Miss Georgette Alexandra's destination is St. Louis, Missouri, US.
  Fact 0.909: Maguire, Mr. John Edward was 30 years old.
  Fact 0.910: Maguire, Mr. John Edward is from Brockton, Massachusetts, US.
  Fact 0.911: Maguire, Mr. John Edward boarded at Southampton.
  Fact 0.912: Maguire, Mr. John Edward's destination is Brockton, Massachusetts, US.
  Fact 0.913: Maréchal, Mr. Pierre, Sr. was 28 years old.
  Fact 0.914: Maréchal, Mr. Pierre, Sr. is from Paris, France.
  Fact 0.915: Maréchal, Mr. Pierre, Sr. boarded at Cherbourg.
  Fact 0.916: Maréchal, Mr. Pierre, Sr.'s destination is New York City.
  Fact 0.917: Maréchal, Mr. Pierre, Sr. has 7 in the lifeboat or body column.
  Fact 0.918: Marvin, Mr. Daniel Warner was 18 years old.
  Fact 0.919: Marvin, Mr. Daniel Warner is from New York City.
  Fact 0.920: Marvin, Mr. Daniel Warner boarded at Southampton.
  Fact 0.921: Marvin, Mr. Daniel Warner's destination is New York City.
  Fact 0.922: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) was 18 years old.
  Fact 0.923: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) is from New York City.
  Fact 0.924: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) boarded at Southampton.
  Fact 0.925: Marvin, Mrs. Mary Graham Carmichael (née Farquarson)'s destination is New York City.
  Fact 0.926: Marvin, Mrs. Mary Graham Carmichael (née Farquarson) has 10 in the lifeboat or body column.
  Fact 0.927: Mayné, Miss Bertha Antonine was 24 years old.
  Fact 0.928: Mayné, Miss Bertha Antonine is from Brussels, Belgium.
  Fact 0.929: Mayné, Miss Bertha Antonine boarded at Cherbourg.
  Fact 0.930: Mayné, Miss Bertha Antonine's destination is Montreal, Quebec, Canada.
  Fact 0.931: Mayné, Miss Bertha Antonine has 6 in the lifeboat or body column.
  Fact 0.932: McCaffry, Mr. Thomas Francis was 46 years old.
  Fact 0.933: McCaffry, Mr. Thomas Francis is from Vancouver, British Columbia, Canada.
  Fact 0.934: McCaffry, Mr. Thomas Francis boarded at Southampton.
  Fact 0.935: McCaffry, Mr. Thomas Francis's destination is Vancouver, British Columbia, Canada.
  Fact 0.936: McCaffry, Mr. Thomas Francis has 292 in the lifeboat or body column.
  Fact 0.937: McCarthy, Mr. Timothy J. was 54 years old.
  Fact 0.938: McCarthy, Mr. Timothy J. is from Dorchester, Massachusetts, US.
  Fact 0.939: McCarthy, Mr. Timothy J. boarded at Southampton.
  Fact 0.940: McCarthy, Mr. Timothy J.'s destination is Dorchester, Massachusetts, US.
  Fact 0.941: McCarthy, Mr. Timothy J. has 175 in the lifeboat or body column.
  Fact 0.942: McGough, Mr. James Robert was 35 years old.
  Fact 0.943: McGough, Mr. James Robert is from Philadelphia, Pennsylvania, US.
  Fact 0.944: McGough, Mr. James Robert boarded at Southampton.
  Fact 0.945: McGough, Mr. James Robert's destination is Philadelphia, Pennsylvania, US.
  Fact 0.946: McGough, Mr. James Robert has 7 in the lifeboat or body column.
  Fact 0.947: Meyer, Mr. Edgar Joseph was 28 years old.
  Fact 0.948: Meyer, Mr. Edgar Joseph is from New York City.
  Fact 0.949: Meyer, Mr. Edgar Joseph boarded at Cherbourg.
  Fact 0.950: Meyer, Mr. Edgar Joseph's destination is New York City.
  Fact 0.951: Meyer, Mrs. Leila (née Saks) was 25 years old.
  Fact 0.952: Meyer, Mrs. Leila (née Saks) is from New York City.
  Fact 0.953: Meyer, Mrs. Leila (née Saks) boarded at Cherbourg.
  Fact 0.954: Meyer, Mrs. Leila (née Saks)'s destination is New York City.
  Fact 0.955: Meyer, Mrs. Leila (née Saks) has 6 in the lifeboat or body column.
  Fact 0.956: Millet, Mr. Francis Davis was 65 years old.
  Fact 0.957: Millet, Mr. Francis Davis is from East Bridgewater, Massachusetts, US.
  Fact 0.958: Millet, Mr. Francis Davis boarded at Cherbourg.
  Fact 0.959: Millet, Mr. Francis Davis's destination is East Bridgewater, Massachusetts, US.
  Fact 0.960: Millet, Mr. Francis Davis has 249 in the lifeboat or body column.
  Fact 0.961: Minahan, Dr. William Edward was 44 years old.
  Fact 0.962: Minahan, Dr. William Edward is from Fond du Lac, Wisconsin, US.
  Fact 0.963: Minahan, Dr. William Edward boarded at Southampton.
  Fact 0.964: Minahan, Dr. William Edward's destination is Fond du Lac, Wisconsin, US.
  Fact 0.965: Minahan, Dr. William Edward has 230 in the lifeboat or body column.
  Fact 0.966: Minahan, Mrs. Lillian E. (née Thorpe) was 37 years old.
  Fact 0.967: Minahan, Mrs. Lillian E. (née Thorpe) is from Fond du Lac, Wisconsin, US.
  Fact 0.968: Minahan, Mrs. Lillian E. (née Thorpe) boarded at Southampton.
  Fact 0.969: Minahan, Mrs. Lillian E. (née Thorpe)'s destination is Fond du Lac, Wisconsin, US.
  Fact 0.970: Minahan, Mrs. Lillian E. (née Thorpe) has 14 in the lifeboat or body column.
  Fact 0.971: Minahan, Miss Daisy E. was 33 years old.
  Fact 0.972: Minahan, Miss Daisy E. is from Fond du Lac, Wisconsin, US.
  Fact 0.973: Minahan, Miss Daisy E. boarded at Southampton.
  Fact 0.974: Minahan, Miss Daisy E.'s destination is Fond du Lac, Wisconsin, US.
  Fact 0.975: Minahan, Miss Daisy E. has 14 in the lifeboat or body column.
  Fact 0.976: Mock, Mr. Philipp Edmund was 30 years old.
  Fact 0.977: Mock, Mr. Philipp Edmund is from New York City.
  Fact 0.978: Mock, Mr. Philipp Edmund boarded at Cherbourg.
  Fact 0.979: Mock, Mr. Philipp Edmund's destination is New York City.
  Fact 0.980: Mock, Mr. Philipp Edmund has 11 in the lifeboat or body column.
  Fact 0.981: Molson, Mr. Harry Markland was 55 years old.
  Fact 0.982: Molson, Mr. Harry Markland is from Montreal, Quebec, Canada.
  Fact 0.983: Molson, Mr. Harry Markland boarded at Southampton.
  Fact 0.984: Molson, Mr. Harry Markland's destination is Montreal, Quebec, Canada.
  Fact 0.985: Moore, Mr. Clarence Bloomfield was 47 years old.
  Fact 0.986: Moore, Mr. Clarence Bloomfield is from Washington, D.C., US.
  Fact 0.987: Moore, Mr. Clarence Bloomfield boarded at Southampton.
  Fact 0.988: Moore, Mr. Clarence Bloomfield's destination is Washington, D.C., US.
  Fact 0.989: Mr. Charles Henry Harrington was 37 years old.
  Fact 0.990: Mr. Charles Henry Harrington is listed as the valet.
  Fact 0.991: Mr. Charles Henry Harrington is from Washington, D.C., US.
  Fact 0.992: Mr. Charles Henry Harrington boarded at Southampton.
  Fact 0.993: Mr. Charles Henry Harrington's destination is Washington, D.C., US.
  Fact 0.994: Natsch, Mr. Charles was 36 years old.
  Fact 0.995: Natsch, Mr. Charles is from Brooklyn, New York, US.
  Fact 0.996: Natsch, Mr. Charles boarded at Cherbourg.
  Fact 0.997: Natsch, Mr. Charles's destination is Brooklyn, New York, US.
  Fact 0.998: Newell, Mr. Arthur Webster was 58 years old.
  Fact 0.999: Newell, Mr. Arthur Webster is from Lexington, Massachusetts, US.
  Fact 0.1000: Newell, Mr. Arthur Webster boarded at Cherbourg.
  Fact 0.1001: Newell, Mr. Arthur Webster's destination is Lexington, Massachusetts, US.
  Fact 0.1002: Newell, Mr. Arthur Webster has 122 in the lifeboat or body column.
  Fact 0.1003: Newell, Miss Madeleine was 31 years old.
  Fact 0.1004: Newell, Miss Madeleine is from Lexington, Massachusetts, US.
  Fact 0.1005: Newell, Miss Madeleine boarded at Cherbourg.
  Fact 0.1006: Newell, Miss Madeleine's destination is Lexington, Massachusetts, US.
  Fact 0.1007: Newell, Miss Madeleine has 6 in the lifeboat or body column.
  Fact 0.1008: Newell, Miss Marjorie Anne was 23 years old.
  Fact 0.1009: Newell, Miss Marjorie Anne is from Lexington, Massachusetts, US.
  Fact 0.1010: Newell, Miss Marjorie Anne boarded at Cherbourg.
  Fact 0.1011: Newell, Miss Marjorie Anne's destination is Lexington, Massachusetts, US.
  Fact 0.1012: Newell, Miss Marjorie Anne has 6 in the lifeboat or body column.
  Fact 0.1013: Newsom, Miss Helen Monypeny was 19 years old.
  Fact 0.1014: Newsom, Miss Helen Monypeny is from New York City.
  Fact 0.1015: Newsom, Miss Helen Monypeny boarded at Southampton.
  Fact 0.1016: Newsom, Miss Helen Monypeny's destination is New York City.
  Fact 0.1017: Newsom, Miss Helen Monypeny has 5 in the lifeboat or body column.
  Fact 0.1018: Nicholson, Mr. Arthur Ernest was 64 years old.
  Fact 0.1019: Nicholson, Mr. Arthur Ernest is from Shanklin, Isle of Wight, England, UK.
  Fact 0.1020: Nicholson, Mr. Arthur Ernest boarded at Southampton.
  Fact 0.1021: Nicholson, Mr. Arthur Ernest's destination is New York City.
  Fact 0.1022: Nicholson, Mr. Arthur Ernest has 263 in the lifeboat or body column.
  Fact 0.1023: Nourney, Mr. Alfred was 20 years old.
  Fact 0.1024: Nourney, Mr. Alfred is from Cologne, German Empire.
  Fact 0.1025: Nourney, Mr. Alfred boarded at Cherbourg.
  Fact 0.1026: Nourney, Mr. Alfred's destination is New York City.
  Fact 0.1027: Nourney, Mr. Alfred has 7 in the lifeboat or body column.
  Fact 0.1028: Omont, Mr. Alfred Fernand was 29 years old.
  Fact 0.1029: Omont, Mr. Alfred Fernand is from Le Havre, France.
  Fact 0.1030: Omont, Mr. Alfred Fernand boarded at Cherbourg.
  Fact 0.1031: Omont, Mr. Alfred Fernand's destination is New York City.
  Fact 0.1032: Omont, Mr. Alfred Fernand has 7 in the lifeboat or body column.
  Fact 0.1033: Ostby, Mr. Engelhart Cornelius was 64 years old.
  Fact 0.1034: Ostby, Mr. Engelhart Cornelius is from Providence, Rhode Island, US.
  Fact 0.1035: Ostby, Mr. Engelhart Cornelius boarded at Southampton.
  Fact 0.1036: Ostby, Mr. Engelhart Cornelius's destination is Providence, Rhode Island, US.
  Fact 0.1037: Ostby, Mr. Engelhart Cornelius has 234 in the lifeboat or body column.
  Fact 0.1038: Ostby, Miss Helene Ragnhild was 22 years old.
  Fact 0.1039: Ostby, Miss Helene Ragnhild is from Providence, Rhode Island, US.
  Fact 0.1040: Ostby, Miss Helene Ragnhild boarded at Southampton.
  Fact 0.1041: Ostby, Miss Helene Ragnhild's destination is Providence, Rhode Island, US.
  Fact 0.1042: Ostby, Miss Helene Ragnhild has 5 in the lifeboat or body column.
  Fact 0.1043: Ovies y Rodriguez, Mr. Servando José Florentino was 36 years old.
  Fact 0.1044: Ovies y Rodriguez, Mr. Servando José Florentino is from Havana, Cuba.
  Fact 0.1045: Ovies y Rodriguez, Mr. Servando José Florentino boarded at Cherbourg.
  Fact 0.1046: Ovies y Rodriguez, Mr. Servando José Florentino's destination is Havana, Cuba.
  Fact 0.1047: Ovies y Rodriguez, Mr. Servando José Florentino has 189 in the lifeboat or body column.
  Fact 0.1048: Parr, Mr. William Henry Marsh was 29 years old.
  Fact 0.1049: Parr, Mr. William Henry Marsh is from Belfast, Ireland, UK.
  Fact 0.1050: Parr, Mr. William Henry Marsh boarded at Belfast.
  Fact 0.1051: Parr, Mr. William Henry Marsh's destination is New York City.
  Fact 0.1052: Partner, Mr. Austin was 40 years old.
  Fact 0.1053: Partner, Mr. Austin is from Tolworth, London, England, UK.
  Fact 0.1054: Partner, Mr. Austin boarded at Southampton.
  Fact 0.1055: Partner, Mr. Austin's destination is Toronto, Ontario, Canada.
  Fact 0.1056: Partner, Mr. Austin has 166 in the lifeboat or body column.
  Fact 0.1057: Pears, Mr. Thomas Clinton was 29 years old.
  Fact 0.1058: Pears, Mr. Thomas Clinton is from Isleworth, London, England, UK.
  Fact 0.1059: Pears, Mr. Thomas Clinton boarded at Southampton.
  Fact 0.1060: Pears, Mr. Thomas Clinton's destination is New York City.
  Fact 0.1061: Pears, Mrs. Edith (née Wearne) was 22 years old.
  Fact 0.1062: Pears, Mrs. Edith (née Wearne) is from Isleworth, London, England, UK.
  Fact 0.1063: Pears, Mrs. Edith (née Wearne) boarded at Southampton.
  Fact 0.1064: Pears, Mrs. Edith (née Wearne)'s destination is New York City.
  Fact 0.1065: Pears, Mrs. Edith (née Wearne) has 8 in the lifeboat or body column.
  Fact 0.1066: Peñasco y Castellana, Mr. Victor was 24 years old.
  Fact 0.1067: Peñasco y Castellana, Mr. Victor is from Madrid, Spain.
  Fact 0.1068: Peñasco y Castellana, Mr. Victor boarded at Cherbourg.
  Fact 0.1069: Peñasco y Castellana, Mr. Victor's destination is New York City.
  Fact 0.1070: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) was 22 years old.
  Fact 0.1071: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) is from Madrid, Spain.
  Fact 0.1072: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) boarded at Cherbourg.
  Fact 0.1073: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo)'s destination is New York City.
  Fact 0.1074: Peñasco y Castellana, Mrs. Maria Josefa (née Perez de Soto y Vallejo) has 8 in the lifeboat or body column.
  Fact 0.1075: Doña Fermina Oliva y Ocana was 39 years old.
  Fact 0.1076: Doña Fermina Oliva y Ocana is listed as the maid.
  Fact 0.1077: Doña Fermina Oliva y Ocana is from Madrid, Spain.
  Fact 0.1078: Doña Fermina Oliva y Ocana boarded at Cherbourg.
  Fact 0.1079: Doña Fermina Oliva y Ocana's destination is New York City.
  Fact 0.1080: Doña Fermina Oliva y Ocana has 8 in the lifeboat or body column.
  Fact 0.1081: Peuchen, Major Arthur Godfrey was 52 years old.
  Fact 0.1082: Peuchen, Major Arthur Godfrey is from Toronto, Ontario, Canada.
  Fact 0.1083: Peuchen, Major Arthur Godfrey boarded at Southampton.
  Fact 0.1084: Peuchen, Major Arthur Godfrey's destination is Toronto, Ontario, Canada.
  Fact 0.1085: Peuchen, Major Arthur Godfrey has 6 in the lifeboat or body column.
  Fact 0.1086: Porter, Mr. Walter Chamberlain was 46 years old.
  Fact 0.1087: Porter, Mr. Walter Chamberlain is from Worcester, Massachusetts, US.
  Fact 0.1088: Porter, Mr. Walter Chamberlain boarded at Southampton.
  Fact 0.1089: Porter, Mr. Walter Chamberlain's destination is Worcester, Massachusetts, US.
  Fact 0.1090: Porter, Mr. Walter Chamberlain has 207 in the lifeboat or body column.
  Fact 0.1091: Potter, Mrs. Lily Alexenia (née Wilson) was 56 years old.
  Fact 0.1092: Potter, Mrs. Lily Alexenia (née Wilson) is from Philadelphia, Pennsylvania, US.
  Fact 0.1093: Potter, Mrs. Lily Alexenia (née Wilson) boarded at Cherbourg.
  Fact 0.1094: Potter, Mrs. Lily Alexenia (née Wilson)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.1095: Potter, Mrs. Lily Alexenia (née Wilson) has 7 in the lifeboat or body column.
  Fact 0.1096: Reuchlin, The Honourable Mr. Johan George was 38 years old.
  Fact 0.1097: Reuchlin, The Honourable Mr. Johan George is from Rotterdam, The Netherlands.
  Fact 0.1098: Reuchlin, The Honourable Mr. Johan George boarded at Southampton.
  Fact 0.1099: Reuchlin, The Honourable Mr. Johan George's destination is New York City.
  Fact 0.1100: Rheims, Mr. George Alexander Lucien was 36 years old.
  Fact 0.1101: Rheims, Mr. George Alexander Lucien is from Paris, France.
  Fact 0.1102: Rheims, Mr. George Alexander Lucien boarded at Southampton.
  Fact 0.1103: Rheims, Mr. George Alexander Lucien's destination is New York City.
  Fact 0.1104: Robert, Mrs. Elisabeth Walton (née McMillan) was 43 years old.
  Fact 0.1105: Robert, Mrs. Elisabeth Walton (née McMillan) is from St. Louis, Missouri, US.
  Fact 0.1106: Robert, Mrs. Elisabeth Walton (née McMillan) boarded at Southampton.
  Fact 0.1107: Robert, Mrs. Elisabeth Walton (née McMillan)'s destination is St. Louis, Missouri, US.
  Fact 0.1108: Miss Emilie Kreuchen was 29 years old.
  Fact 0.1109: Miss Emilie Kreuchen is listed as the maid.
  Fact 0.1110: Miss Emilie Kreuchen is from St. Louis, Missouri, US.
  Fact 0.1111: Miss Emilie Kreuchen boarded at Southampton.
  Fact 0.1112: Miss Emilie Kreuchen's destination is St. Louis, Missouri, US.
  Fact 0.1113: Roebling, Mr. Washington Augustus II was 31 years old.
  Fact 0.1114: Roebling, Mr. Washington Augustus II is from Trenton, New Jersey, US.
  Fact 0.1115: Roebling, Mr. Washington Augustus II boarded at Southampton.
  Fact 0.1116: Roebling, Mr. Washington Augustus II's destination is Trenton, New Jersey, US.
  Fact 0.1117: Romaine, Mr. Charles Hallance (alias C. Rolmane) was 45 years old.
  Fact 0.1118: Romaine, Mr. Charles Hallance (alias C. Rolmane) is from New York City.
  Fact 0.1119: Romaine, Mr. Charles Hallance (alias C. Rolmane) boarded at Southampton.
  Fact 0.1120: Romaine, Mr. Charles Hallance (alias C. Rolmane)'s destination is New York City.
  Fact 0.1121: Romaine, Mr. Charles Hallance (alias C. Rolmane) has 9 in the lifeboat or body column.
  Fact 0.1122: Rood, Mr. Hugh Roscoe was 38 years old.
  Fact 0.1123: Rood, Mr. Hugh Roscoe is from Seattle, Washington, US.
  Fact 0.1124: Rood, Mr. Hugh Roscoe boarded at Southampton.
  Fact 0.1125: Rood, Mr. Hugh Roscoe's destination is Seattle, Washington, US.
  Fact 0.1126: Rosenbaum, Miss Edith Louise was 34 years old.
  Fact 0.1127: Rosenbaum, Miss Edith Louise is from Paris, France.
  Fact 0.1128: Rosenbaum, Miss Edith Louise boarded at Cherbourg.
  Fact 0.1129: Rosenbaum, Miss Edith Louise's destination is Cincinnati, Ohio, US.
  Fact 0.1130: Rosenbaum, Miss Edith Louise has 11 in the lifeboat or body column.
  Fact 0.1131: Rosenshine, Mr. George (alias George Thorne) was 46 years old.
  Fact 0.1132: Rosenshine, Mr. George (alias George Thorne) is from New York City.
  Fact 0.1133: Rosenshine, Mr. George (alias George Thorne) boarded at Cherbourg.
  Fact 0.1134: Rosenshine, Mr. George (alias George Thorne)'s destination is New York City.
  Fact 0.1135: Rosenshine, Mr. George (alias George Thorne) has 16 in the lifeboat or body column.
  Fact 0.1136: Ross, Mr. John Hugo was 36 years old.
  Fact 0.1137: Ross, Mr. John Hugo is from Winnipeg, Manitoba, Canada.
  Fact 0.1138: Ross, Mr. John Hugo boarded at Southampton.
  Fact 0.1139: Ross, Mr. John Hugo's destination is Winnipeg, Manitoba, Canada.
  Fact 0.1140: Rothschild, Mr. Martin was 46 years old.
  Fact 0.1141: Rothschild, Mr. Martin is from New York City.
  Fact 0.1142: Rothschild, Mr. Martin boarded at Cherbourg.
  Fact 0.1143: Rothschild, Mr. Martin's destination is New York City.
  Fact 0.1144: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) was 54 years old.
  Fact 0.1145: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) is from New York City.
  Fact 0.1146: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) boarded at Cherbourg.
  Fact 0.1147: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett)'s destination is New York City.
  Fact 0.1148: Rothschild, Mrs. Elizabeth Jane Anne (née Barrett) has 6 in the lifeboat or body column.
  Fact 0.1149: Rowe, Mr. Alfred G. was 59 years old.
  Fact 0.1150: Rowe, Mr. Alfred G. is from Liverpool, Merseyside, England, UK.
  Fact 0.1151: Rowe, Mr. Alfred G. boarded at Southampton.
  Fact 0.1152: Rowe, Mr. Alfred G.'s destination is New York City.
  Fact 0.1153: Rowe, Mr. Alfred G. has 109 in the lifeboat or body column.
  Fact 0.1154: Ryerson, Mr. Arthur Larned was 61 years old.
  Fact 0.1155: Ryerson, Mr. Arthur Larned is from Cooperstown, New York, US.
  Fact 0.1156: Ryerson, Mr. Arthur Larned boarded at Cherbourg.
  Fact 0.1157: Ryerson, Mr. Arthur Larned's destination is Cooperstown, New York, US.
  Fact 0.1158: Ryerson, Mrs. Emily Maria (née Borie) was 48 years old.
  Fact 0.1159: Ryerson, Mrs. Emily Maria (née Borie) is from Cooperstown, New York, US.
  Fact 0.1160: Ryerson, Mrs. Emily Maria (née Borie) boarded at Cherbourg.
  Fact 0.1161: Ryerson, Mrs. Emily Maria (née Borie)'s destination is Cooperstown, New York, US.
  Fact 0.1162: Miss Victorine Chaudanson was 36 years old.
  Fact 0.1163: Miss Victorine Chaudanson is listed as the maid.
  Fact 0.1164: Miss Victorine Chaudanson is from Cooperstown, New York, US.
  Fact 0.1165: Miss Victorine Chaudanson boarded at Cherbourg.
  Fact 0.1166: Miss Victorine Chaudanson's destination is Cooperstown, New York, US.
  Fact 0.1167: Ryerson, Miss Susan Parker "Suzette" was 21 years old.
  Fact 0.1168: Ryerson, Miss Susan Parker "Suzette" is from Cooperstown, New York, US.
  Fact 0.1169: Ryerson, Miss Susan Parker "Suzette" boarded at Cherbourg.
  Fact 0.1170: Ryerson, Miss Susan Parker "Suzette"'s destination is Cooperstown, New York, US.
  Fact 0.1171: Ryerson, Miss Emily Borie was 18 years old.
  Fact 0.1172: Ryerson, Miss Emily Borie is from Cooperstown, New York, US.
  Fact 0.1173: Ryerson, Miss Emily Borie boarded at Cherbourg.
  Fact 0.1174: Ryerson, Miss Emily Borie's destination is Cooperstown, New York, US.
  Fact 0.1175: Ryerson, Master John Borie "Jack" was 13 years old.
  Fact 0.1176: Ryerson, Master John Borie "Jack" is from Cooperstown, New York, US.
  Fact 0.1177: Ryerson, Master John Borie "Jack" boarded at Cherbourg.
  Fact 0.1178: Ryerson, Master John Borie "Jack"'s destination is Cooperstown, New York, US.
  Fact 0.1179: Miss Grace Scott Bowen was 45 years old.
  Fact 0.1180: Miss Grace Scott Bowen is listed as the governess.
  Fact 0.1181: Miss Grace Scott Bowen is from Cooperstown, New York, US.
  Fact 0.1182: Miss Grace Scott Bowen boarded at Cherbourg.
  Fact 0.1183: Miss Grace Scott Bowen's destination is Cooperstown, New York, US.
  Fact 0.1184: Saalfeld, Mr. Adolphe was 47 years old.
  Fact 0.1185: Saalfeld, Mr. Adolphe is from Manchester, England, UK.
  Fact 0.1186: Saalfeld, Mr. Adolphe boarded at Southampton.
  Fact 0.1187: Saalfeld, Mr. Adolphe's destination is New York City.
  Fact 0.1188: Salomon, Mr. Abraham Lincoln was 43 years old.
  Fact 0.1189: Salomon, Mr. Abraham Lincoln is from New York City.
  Fact 0.1190: Salomon, Mr. Abraham Lincoln boarded at Cherbourg.
  Fact 0.1191: Salomon, Mr. Abraham Lincoln's destination is New York City.
  Fact 0.1192: Schabert, Mrs. Emma (née Mock) was 35 years old.
  Fact 0.1193: Schabert, Mrs. Emma (née Mock) is from Hamburg, German Empire.
  Fact 0.1194: Schabert, Mrs. Emma (née Mock) boarded at Southampton.
  Fact 0.1195: Schabert, Mrs. Emma (née Mock)'s destination is New York City.
  Fact 0.1196: Schabert, Mrs. Emma (née Mock) has 11 in the lifeboat or body column.
  Fact 0.1197: Seward, Mr. Frederic Kimber was 34 years old.
  Fact 0.1198: Seward, Mr. Frederic Kimber is from New York City.
  Fact 0.1199: Seward, Mr. Frederic Kimber boarded at Southampton.
  Fact 0.1200: Seward, Mr. Frederic Kimber's destination is New York City.
  Fact 0.1201: Seward, Mr. Frederic Kimber has 7 in the lifeboat or body column.
  Fact 0.1202: Silverthorne, Mr. Spencer Victor was 35 years old.
  Fact 0.1203: Silverthorne, Mr. Spencer Victor is from St. Louis, Missouri, US.
  Fact 0.1204: Silverthorne, Mr. Spencer Victor boarded at Southampton.
  Fact 0.1205: Silverthorne, Mr. Spencer Victor's destination is St. Louis, Missouri, US.
  Fact 0.1206: Silverthorne, Mr. Spencer Victor has 5 in the lifeboat or body column.
  Fact 0.1207: Silvey, Mr. William Baird was 50 years old.
  Fact 0.1208: Silvey, Mr. William Baird is from Duluth, Minnesota, US.
  Fact 0.1209: Silvey, Mr. William Baird boarded at Cherbourg.
  Fact 0.1210: Silvey, Mr. William Baird's destination is Duluth, Minnesota, US.
  Fact 0.1211: Silvey, Mrs. Alice (née Munger) was 39 years old.
  Fact 0.1212: Silvey, Mrs. Alice (née Munger) is from Duluth, Minnesota, US.
  Fact 0.1213: Silvey, Mrs. Alice (née Munger) boarded at Cherbourg.
  Fact 0.1214: Silvey, Mrs. Alice (née Munger)'s destination is Duluth, Minnesota, US.
  Fact 0.1215: Silvey, Mrs. Alice (née Munger) has 11 in the lifeboat or body column.
  Fact 0.1216: Simonius-Blumer, Colonel Alfons was 56 years old.
  Fact 0.1217: Simonius-Blumer, Colonel Alfons is from Basel, Switzerland.
  Fact 0.1218: Simonius-Blumer, Colonel Alfons boarded at Southampton.
  Fact 0.1219: Simonius-Blumer, Colonel Alfons's destination is New York City.
  Fact 0.1220: Sloper, Mr. William Thompson was 28 years old.
  Fact 0.1221: Sloper, Mr. William Thompson is from New Britain, Connecticut, US.
  Fact 0.1222: Sloper, Mr. William Thompson boarded at Southampton.
  Fact 0.1223: Sloper, Mr. William Thompson's destination is New Britain, Connecticut, US.
  Fact 0.1224: Sloper, Mr. William Thompson has 7 in the lifeboat or body column.
  Fact 0.1225: Smart, Mr. John Montgomery was 56 years old.
  Fact 0.1226: Smart, Mr. John Montgomery is from Kildale, North Yorkshire, England, UK.
  Fact 0.1227: Smart, Mr. John Montgomery boarded at Southampton.
  Fact 0.1228: Smart, Mr. John Montgomery's destination is New York City.
  Fact 0.1229: Smith, Mr. James Clinch was 56 years old.
  Fact 0.1230: Smith, Mr. James Clinch is from Paris, France.
  Fact 0.1231: Smith, Mr. James Clinch boarded at Cherbourg.
  Fact 0.1232: Smith, Mr. James Clinch's destination is Long Island, New York, US.
  Fact 0.1233: Smith, Mr. Lucien Philip was 24 years old.
  Fact 0.1234: Smith, Mr. Lucien Philip is from Huntington, West Virginia, US.
  Fact 0.1235: Smith, Mr. Lucien Philip boarded at Cherbourg.
  Fact 0.1236: Smith, Mr. Lucien Philip's destination is Huntington, West Virginia, US.
  Fact 0.1237: Smith, Mrs. Mary Eloise (née Hughes) was 18 years old.
  Fact 0.1238: Smith, Mrs. Mary Eloise (née Hughes) is from Huntington, West Virginia, US.
  Fact 0.1239: Smith, Mrs. Mary Eloise (née Hughes) boarded at Cherbourg.
  Fact 0.1240: Smith, Mrs. Mary Eloise (née Hughes)'s destination is Huntington, West Virginia, US.
  Fact 0.1241: Smith, Mrs. Mary Eloise (née Hughes) has 6 in the lifeboat or body column.
  Fact 0.1242: Smith, Mr. Richard William was 57 years old.
  Fact 0.1243: Smith, Mr. Richard William is from Streatham, London, England, UK.
  Fact 0.1244: Smith, Mr. Richard William boarded at Southampton.
  Fact 0.1245: Smith, Mr. Richard William's destination is New York City.
  Fact 0.1246: Snyder, Mr. John Pillsbury was 24 years old.
  Fact 0.1247: Snyder, Mr. John Pillsbury is from Minneapolis, Minnesota, US.
  Fact 0.1248: Snyder, Mr. John Pillsbury boarded at Southampton.
  Fact 0.1249: Snyder, Mr. John Pillsbury's destination is Minneapolis, Minnesota, US.
  Fact 0.1250: Snyder, Mr. John Pillsbury has 7 in the lifeboat or body column.
  Fact 0.1251: Snyder, Mrs. Nellie (née Stevenson) was 23 years old.
  Fact 0.1252: Snyder, Mrs. Nellie (née Stevenson) is from Minneapolis, Minnesota, US.
  Fact 0.1253: Snyder, Mrs. Nellie (née Stevenson) boarded at Southampton.
  Fact 0.1254: Snyder, Mrs. Nellie (née Stevenson)'s destination is Minneapolis, Minnesota, US.
  Fact 0.1255: Snyder, Mrs. Nellie (née Stevenson) has 7 in the lifeboat or body column.
  Fact 0.1256: Spedden, Mr. Frederic Oakley was 45 years old.
  Fact 0.1257: Spedden, Mr. Frederic Oakley is from Tuxedo Park, New York, US.
  Fact 0.1258: Spedden, Mr. Frederic Oakley boarded at Cherbourg.
  Fact 0.1259: Spedden, Mr. Frederic Oakley's destination is Tuxedo Park, New York, US.
  Fact 0.1260: Spedden, Mrs. Margaretta Corning (née Stone) was 39 years old.
  Fact 0.1261: Spedden, Mrs. Margaretta Corning (née Stone) is from Tuxedo Park, New York, US.
  Fact 0.1262: Spedden, Mrs. Margaretta Corning (née Stone) boarded at Cherbourg.
  Fact 0.1263: Spedden, Mrs. Margaretta Corning (née Stone)'s destination is Tuxedo Park, New York, US.
  Fact 0.1264: Miss Helen Alice Wilson was 31 years old.
  Fact 0.1265: Miss Helen Alice Wilson is listed as the maid.
  Fact 0.1266: Miss Helen Alice Wilson is from Tuxedo Park, New York, US.
  Fact 0.1267: Miss Helen Alice Wilson boarded at Cherbourg.
  Fact 0.1268: Miss Helen Alice Wilson's destination is Tuxedo Park, New York, US.
  Fact 0.1269: Spedden, Master Robert Douglas was 6 years old.
  Fact 0.1270: Spedden, Master Robert Douglas is from Tuxedo Park, New York, US.
  Fact 0.1271: Spedden, Master Robert Douglas boarded at Cherbourg.
  Fact 0.1272: Spedden, Master Robert Douglas's destination is Tuxedo Park, New York, US.
  Fact 0.1273: Miss Elizabeth Margaret Burns was 41 years old.
  Fact 0.1274: Miss Elizabeth Margaret Burns is listed as the nurse.
  Fact 0.1275: Miss Elizabeth Margaret Burns is from Tuxedo Park, New York, US.
  Fact 0.1276: Miss Elizabeth Margaret Burns boarded at Cherbourg.
  Fact 0.1277: Miss Elizabeth Margaret Burns's destination is Tuxedo Park, New York, US.
  Fact 0.1278: Spencer, Mr. William Augustus was 57 years old.
  Fact 0.1279: Spencer, Mr. William Augustus is from New York City.
  Fact 0.1280: Spencer, Mr. William Augustus boarded at Cherbourg.
  Fact 0.1281: Spencer, Mr. William Augustus's destination is New York City.
  Fact 0.1282: Spencer, Mrs. Marie Eugénie (née Demougeot) was 45 years old.
  Fact 0.1283: Spencer, Mrs. Marie Eugénie (née Demougeot) is from New York City.
  Fact 0.1284: Spencer, Mrs. Marie Eugénie (née Demougeot) boarded at Cherbourg.
  Fact 0.1285: Spencer, Mrs. Marie Eugénie (née Demougeot)'s destination is New York City.
  Fact 0.1286: Spencer, Mrs. Marie Eugénie (née Demougeot) has 6 in the lifeboat or body column.
  Fact 0.1287: Miss Eugenie Elise Lurette was 59 years old.
  Fact 0.1288: Miss Eugenie Elise Lurette is listed as the maid.
  Fact 0.1289: Miss Eugenie Elise Lurette is from New York City.
  Fact 0.1290: Miss Eugenie Elise Lurette boarded at Cherbourg.
  Fact 0.1291: Miss Eugenie Elise Lurette's destination is New York City.
  Fact 0.1292: Miss Eugenie Elise Lurette has 6 in the lifeboat or body column.
  Fact 0.1293: Stähelin-Maeglin, Dr. Max was 32 years old.
  Fact 0.1294: Stähelin-Maeglin, Dr. Max is from Basel, Switzerland.
  Fact 0.1295: Stähelin-Maeglin, Dr. Max boarded at Southampton.
  Fact 0.1296: Stähelin-Maeglin, Dr. Max's destination is New York City.
  Fact 0.1297: Stead, Mr. William Thomas was 62 years old.
  Fact 0.1298: Stead, Mr. William Thomas is from London, England, UK.
  Fact 0.1299: Stead, Mr. William Thomas boarded at Southampton.
  Fact 0.1300: Stead, Mr. William Thomas's destination is New York City.
  Fact 0.1301: Stengel, Mr. Charles Emil Henry was 54 years old.
  Fact 0.1302: Stengel, Mr. Charles Emil Henry is from Newark, New Jersey, US.
  Fact 0.1303: Stengel, Mr. Charles Emil Henry boarded at Cherbourg.
  Fact 0.1304: Stengel, Mr. Charles Emil Henry's destination is Newark, New Jersey, US.
  Fact 0.1305: Stengel, Mrs. Annie May (née Morris) was 44 years old.
  Fact 0.1306: Stengel, Mrs. Annie May (née Morris) is from Newark, New Jersey, US.
  Fact 0.1307: Stengel, Mrs. Annie May (née Morris) boarded at Cherbourg.
  Fact 0.1308: Stengel, Mrs. Annie May (née Morris)'s destination is Newark, New Jersey, US.
  Fact 0.1309: Stengel, Mrs. Annie May (née Morris) has 5 in the lifeboat or body column.
  Fact 0.1310: Stephenson, Mrs. Martha (née Eustis) was 52 years old.
  Fact 0.1311: Stephenson, Mrs. Martha (née Eustis) is from Paris, France.
  Fact 0.1312: Stephenson, Mrs. Martha (née Eustis) boarded at Cherbourg.
  Fact 0.1313: Stephenson, Mrs. Martha (née Eustis)'s destination is Haverford, Pennsylvania, US.
  Fact 0.1314: Stewart, Mr. Albert A. was 54 years old.
  Fact 0.1315: Stewart, Mr. Albert A. is from Cincinnati, Ohio, US.
  Fact 0.1316: Stewart, Mr. Albert A. boarded at Cherbourg.
  Fact 0.1317: Stewart, Mr. Albert A.'s destination is Gallipolis, Ohio, US.
  Fact 0.1318: Stone, Mrs. Martha Evelyn (née Stevens) was 62 years old.
  Fact 0.1319: Stone, Mrs. Martha Evelyn (née Stevens) is from New York City.
  Fact 0.1320: Stone, Mrs. Martha Evelyn (née Stevens) boarded at Southampton.
  Fact 0.1321: Stone, Mrs. Martha Evelyn (née Stevens)'s destination is New York City.
  Fact 0.1322: Stone, Mrs. Martha Evelyn (née Stevens) has 6 in the lifeboat or body column.
  Fact 0.1323: Miss Amelie "Amelia" Icard was 38 years old.
  Fact 0.1324: Miss Amelie "Amelia" Icard is listed as the maid.
  Fact 0.1325: Miss Amelie "Amelia" Icard is from New York City.
  Fact 0.1326: Miss Amelie "Amelia" Icard boarded at Southampton.
  Fact 0.1327: Miss Amelie "Amelia" Icard's destination is New York City.
  Fact 0.1328: Miss Amelie "Amelia" Icard has 6 in the lifeboat or body column.
  Fact 0.1329: Straus, Mr. Isidor was 67 years old.
  Fact 0.1330: Straus, Mr. Isidor is from New York City.
  Fact 0.1331: Straus, Mr. Isidor boarded at Southampton.
  Fact 0.1332: Straus, Mr. Isidor's destination is New York City.
  Fact 0.1333: Straus, Mr. Isidor has 96 in the lifeboat or body column.
  Fact 0.1334: Mr. John Farthing was 57 years old.
  Fact 0.1335: Mr. John Farthing is listed as the valet.
  Fact 0.1336: Mr. John Farthing is from New York City.
  Fact 0.1337: Mr. John Farthing boarded at Southampton.
  Fact 0.1338: Mr. John Farthing's destination is New York City.
  Fact 0.1339: Straus, Mrs. Rosalie Ida (née Blun) was 63 years old.
  Fact 0.1340: Straus, Mrs. Rosalie Ida (née Blun) is from New York City.
  Fact 0.1341: Straus, Mrs. Rosalie Ida (née Blun) boarded at Southampton.
  Fact 0.1342: Straus, Mrs. Rosalie Ida (née Blun)'s destination is New York City.
  Fact 0.1343: Miss Ellen Bird was 31 years old.
  Fact 0.1344: Miss Ellen Bird is listed as the maid.
  Fact 0.1345: Miss Ellen Bird is from New York City.
  Fact 0.1346: Miss Ellen Bird boarded at Southampton.
  Fact 0.1347: Miss Ellen Bird's destination is New York City.
  Fact 0.1348: Miss Ellen Bird has 8 in the lifeboat or body column.
  Fact 0.1349: Sutton, Mr. Frederick was 61 years old.
  Fact 0.1350: Sutton, Mr. Frederick is from Haddonfield, New Jersey, US.
  Fact 0.1351: Sutton, Mr. Frederick boarded at Southampton.
  Fact 0.1352: Sutton, Mr. Frederick's destination is Haddonfield, New Jersey, US.
  Fact 0.1353: Sutton, Mr. Frederick has 46 in the lifeboat or body column.
  Fact 0.1354: Swift, Mrs. Margaret Welles (née Barron) was 46 years old.
  Fact 0.1355: Swift, Mrs. Margaret Welles (née Barron) is from New York City.
  Fact 0.1356: Swift, Mrs. Margaret Welles (née Barron) boarded at Southampton.
  Fact 0.1357: Swift, Mrs. Margaret Welles (née Barron)'s destination is New York City.
  Fact 0.1358: Swift, Mrs. Margaret Welles (née Barron) has 8 in the lifeboat or body column.
  Fact 0.1359: Taussig, Mr. Emil was 52 years old.
  Fact 0.1360: Taussig, Mr. Emil is from New York City.
  Fact 0.1361: Taussig, Mr. Emil boarded at Southampton.
  Fact 0.1362: Taussig, Mr. Emil's destination is New York City.
  Fact 0.1363: Taussig, Mrs. Tillie (née Mandelbaum) was 39 years old.
  Fact 0.1364: Taussig, Mrs. Tillie (née Mandelbaum) is from New York City.
  Fact 0.1365: Taussig, Mrs. Tillie (née Mandelbaum) boarded at Southampton.
  Fact 0.1366: Taussig, Mrs. Tillie (née Mandelbaum)'s destination is New York City.
  Fact 0.1367: Taussig, Mrs. Tillie (née Mandelbaum) has 8 in the lifeboat or body column.
  Fact 0.1368: Taussig, Miss Ruth was 18 years old.
  Fact 0.1369: Taussig, Miss Ruth is from New York City.
  Fact 0.1370: Taussig, Miss Ruth boarded at Southampton.
  Fact 0.1371: Taussig, Miss Ruth's destination is New York City.
  Fact 0.1372: Taussig, Miss Ruth has 8 in the lifeboat or body column.
  Fact 0.1373: Taylor, Mr. Elmer Zebley was 48 years old.
  Fact 0.1374: Taylor, Mr. Elmer Zebley is from London, England, UK.
  Fact 0.1375: Taylor, Mr. Elmer Zebley boarded at Southampton.
  Fact 0.1376: Taylor, Mr. Elmer Zebley's destination is East Orange, New Jersey, US.
  Fact 0.1377: Taylor, Mr. Elmer Zebley has 5 in the lifeboat or body column.
  Fact 0.1378: Taylor, Mrs. Juliet Cummins (née Wright) was 48 years old.
  Fact 0.1379: Taylor, Mrs. Juliet Cummins (née Wright) is from London, England, UK.
  Fact 0.1380: Taylor, Mrs. Juliet Cummins (née Wright) boarded at Southampton.
  Fact 0.1381: Taylor, Mrs. Juliet Cummins (née Wright)'s destination is East Orange, New Jersey, US.
  Fact 0.1382: Taylor, Mrs. Juliet Cummins (née Wright) has 5 in the lifeboat or body column.
  Fact 0.1383: Thayer, Mr. John Borland II was 49 years old.
  Fact 0.1384: Thayer, Mr. John Borland II is from Haverford, Pennsylvania, US.
  Fact 0.1385: Thayer, Mr. John Borland II boarded at Cherbourg.
  Fact 0.1386: Thayer, Mr. John Borland II's destination is Haverford, Pennsylvania, US.
  Fact 0.1387: Thayer, Mrs. Marian Longsteth (née Morris) was 39 years old.
  Fact 0.1388: Thayer, Mrs. Marian Longsteth (née Morris) is from Haverford, Pennsylvania, US.
  Fact 0.1389: Thayer, Mrs. Marian Longsteth (née Morris) boarded at Cherbourg.
  Fact 0.1390: Thayer, Mrs. Marian Longsteth (née Morris)'s destination is Haverford, Pennsylvania, US.
  Fact 0.1391: Miss Margaret Fleming was 42 years old.
  Fact 0.1392: Miss Margaret Fleming is listed as the maid.
  Fact 0.1393: Miss Margaret Fleming is from Haverford, Pennsylvania, US.
  Fact 0.1394: Miss Margaret Fleming boarded at Cherbourg.
  Fact 0.1395: Miss Margaret Fleming's destination is Haverford, Pennsylvania, US.
  Fact 0.1396: Thayer, Mr. John Borland "Jack" III was 17 years old.
  Fact 0.1397: Thayer, Mr. John Borland "Jack" III is from Haverford, Pennsylvania, US.
  Fact 0.1398: Thayer, Mr. John Borland "Jack" III boarded at Cherbourg.
  Fact 0.1399: Thayer, Mr. John Borland "Jack" III's destination is Haverford, Pennsylvania, US.
  Fact 0.1400: Thorne, Mrs. Gertrude Maybelle (née McMinn) was 37 years old.
  Fact 0.1401: Thorne, Mrs. Gertrude Maybelle (née McMinn) is from New York City, New York, US.
  Fact 0.1402: Thorne, Mrs. Gertrude Maybelle (née McMinn) boarded at Cherbourg.
  Fact 0.1403: Thorne, Mrs. Gertrude Maybelle (née McMinn)'s destination is New York City, New York, US.
  Fact 0.1404: Tucker, Mr. Gilbert Milligan Jr. was 31 years old.
  Fact 0.1405: Tucker, Mr. Gilbert Milligan Jr. is from Albany, New York, US.
  Fact 0.1406: Tucker, Mr. Gilbert Milligan Jr. boarded at Cherbourg.
  Fact 0.1407: Tucker, Mr. Gilbert Milligan Jr.'s destination is Albany, New York, US.
  Fact 0.1408: Tucker, Mr. Gilbert Milligan Jr. has 7 in the lifeboat or body column.
  Fact 0.1409: Uruchurtu, Don Manuel E. was 40 years old.
  Fact 0.1410: Uruchurtu, Don Manuel E. is from Mexico City, Mexico.
  Fact 0.1411: Uruchurtu, Don Manuel E. boarded at Cherbourg.
  Fact 0.1412: Uruchurtu, Don Manuel E.'s destination is Mexico City, Mexico.
  Fact 0.1413: Van der Hoef, Mr. Wyckoff was 61 years old.
  Fact 0.1414: Van der Hoef, Mr. Wyckoff is from Brooklyn, New York, US.
  Fact 0.1415: Van der Hoef, Mr. Wyckoff boarded at Belfast.
  Fact 0.1416: Van der Hoef, Mr. Wyckoff's destination is Brooklyn, New York, US.
  Fact 0.1417: Van der Hoef, Mr. Wyckoff has 245 in the lifeboat or body column.
  Fact 0.1418: Walker, Mr. William Anderson was 48 years old.
  Fact 0.1419: Walker, Mr. William Anderson is from East Orange, New Jersey, US.
  Fact 0.1420: Walker, Mr. William Anderson boarded at Southampton.
  Fact 0.1421: Walker, Mr. William Anderson's destination is East Orange, New Jersey, US.
  Fact 0.1422: Warren, Mr. Frank Manley was 63 years old.
  Fact 0.1423: Warren, Mr. Frank Manley is from Portland, Oregon, US.
  Fact 0.1424: Warren, Mr. Frank Manley boarded at Cherbourg.
  Fact 0.1425: Warren, Mr. Frank Manley's destination is Portland, Oregon, US.
  Fact 0.1426: Warren, Mrs. Anna Sophia (née Atkinson) was 60 years old.
  Fact 0.1427: Warren, Mrs. Anna Sophia (née Atkinson) is from Portland, Oregon, US.
  Fact 0.1428: Warren, Mrs. Anna Sophia (née Atkinson) boarded at Cherbourg.
  Fact 0.1429: Warren, Mrs. Anna Sophia (née Atkinson)'s destination is Portland, Oregon, US.
  Fact 0.1430: Warren, Mrs. Anna Sophia (née Atkinson) has 5 in the lifeboat or body column.
  Fact 0.1431: Weir, Colonel John was 59 years old.
  Fact 0.1432: Weir, Colonel John is from New York City.
  Fact 0.1433: Weir, Colonel John boarded at Southampton.
  Fact 0.1434: Weir, Colonel John's destination is New York City.
  Fact 0.1435: White, Mr. Percival Wayland was 54 years old.
  Fact 0.1436: White, Mr. Percival Wayland is from Brunswick, Maine, US.
  Fact 0.1437: White, Mr. Percival Wayland boarded at Southampton.
  Fact 0.1438: White, Mr. Percival Wayland's destination is Brunswick, Maine, US.
  Fact 0.1439: White, Mr. Richard Frasar was 21 years old.
  Fact 0.1440: White, Mr. Richard Frasar is from Brunswick, Maine, US.
  Fact 0.1441: White, Mr. Richard Frasar boarded at Southampton.
  Fact 0.1442: White, Mr. Richard Frasar's destination is Brunswick, Maine, US.
  Fact 0.1443: White, Mr. Richard Frasar has 169 in the lifeboat or body column.
  Fact 0.1444: White, Mrs. Ella (née Holmes) was 55 years old.
  Fact 0.1445: White, Mrs. Ella (née Holmes) is from New York City.
  Fact 0.1446: White, Mrs. Ella (née Holmes) boarded at Cherbourg.
  Fact 0.1447: White, Mrs. Ella (née Holmes)'s destination is New York City.
  Fact 0.1448: White, Mrs. Ella (née Holmes) has 8 in the lifeboat or body column.
  Fact 0.1449: Miss Amelia Mayo "Nellie" Bessette was 39 years old.
  Fact 0.1450: Miss Amelia Mayo "Nellie" Bessette is listed as the maid.
  Fact 0.1451: Miss Amelia Mayo "Nellie" Bessette is from New York City.
  Fact 0.1452: Miss Amelia Mayo "Nellie" Bessette boarded at Cherbourg.
  Fact 0.1453: Miss Amelia Mayo "Nellie" Bessette's destination is New York City.
  Fact 0.1454: Miss Amelia Mayo "Nellie" Bessette has 8 in the lifeboat or body column.
  Fact 0.1455: Mr. Sante Righini was 22 years old.
  Fact 0.1456: Mr. Sante Righini is listed as the manservant.
  Fact 0.1457: Mr. Sante Righini is from New York City.
  Fact 0.1458: Mr. Sante Righini boarded at Cherbourg.
  Fact 0.1459: Mr. Sante Righini's destination is New York City.
  Fact 0.1460: Mr. Sante Righini has 232 in the lifeboat or body column.
  Fact 0.1461: Wick, Colonel George Dennick was 58 years old.
  Fact 0.1462: Wick, Colonel George Dennick is from Youngstown, Ohio, US.
  Fact 0.1463: Wick, Colonel George Dennick boarded at Southampton.
  Fact 0.1464: Wick, Colonel George Dennick's destination is Youngstown, Ohio, US.
  Fact 0.1465: Wick, Colonel George Dennick has 8 in the lifeboat or body column.
  Fact 0.1466: Wick, Mrs. Mary (née Hitchcock) was 45 years old.
  Fact 0.1467: Wick, Mrs. Mary (née Hitchcock) is from Youngstown, Ohio, US.
  Fact 0.1468: Wick, Mrs. Mary (née Hitchcock) boarded at Southampton.
  Fact 0.1469: Wick, Mrs. Mary (née Hitchcock)'s destination is Youngstown, Ohio, US.
  Fact 0.1470: Wick, Mrs. Mary (née Hitchcock) has 8 in the lifeboat or body column.
  Fact 0.1471: Wick, Miss Mary Natalie was 31 years old.
  Fact 0.1472: Wick, Miss Mary Natalie is from Youngstown, Ohio, US.
  Fact 0.1473: Wick, Miss Mary Natalie boarded at Southampton.
  Fact 0.1474: Wick, Miss Mary Natalie's destination is Youngstown, Ohio, US.
  Fact 0.1475: Wick, Miss Mary Natalie has 8 in the lifeboat or body column.
  Fact 0.1476: Widener, Mr. George Dunton was 50 years old.
  Fact 0.1477: Widener, Mr. George Dunton is from Philadelphia, Pennsylvania, US.
  Fact 0.1478: Widener, Mr. George Dunton boarded at Southampton.
  Fact 0.1479: Widener, Mr. George Dunton's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1480: Mr. Edwin Herbert Keeping was 33 years old.
  Fact 0.1481: Mr. Edwin Herbert Keeping is listed as the valet.
  Fact 0.1482: Mr. Edwin Herbert Keeping is from Philadelphia, Pennsylvania, US.
  Fact 0.1483: Mr. Edwin Herbert Keeping boarded at Southampton.
  Fact 0.1484: Mr. Edwin Herbert Keeping's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1485: Mr. Edwin Herbert Keeping has 45 in the lifeboat or body column.
  Fact 0.1486: Widener, Mrs. Eleanor (née Elkins) was 50 years old.
  Fact 0.1487: Widener, Mrs. Eleanor (née Elkins) is from Philadelphia, Pennsylvania, US.
  Fact 0.1488: Widener, Mrs. Eleanor (née Elkins) boarded at Southampton.
  Fact 0.1489: Widener, Mrs. Eleanor (née Elkins)'s destination is Philadelphia, Pennsylvania, US.
  Fact 0.1490: Miss Amalie Henriette "Emily" Gieger was 35 years old.
  Fact 0.1491: Miss Amalie Henriette "Emily" Gieger is listed as the maid.
  Fact 0.1492: Miss Amalie Henriette "Emily" Gieger is from Philadelphia, Pennsylvania, US.
  Fact 0.1493: Miss Amalie Henriette "Emily" Gieger boarded at Southampton.
  Fact 0.1494: Miss Amalie Henriette "Emily" Gieger's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1495: Widener, Mr. Harry Elkins was 27 years old.
  Fact 0.1496: Widener, Mr. Harry Elkins is from Philadelphia, Pennsylvania, US.
  Fact 0.1497: Widener, Mr. Harry Elkins boarded at Southampton.
  Fact 0.1498: Widener, Mr. Harry Elkins's destination is Philadelphia, Pennsylvania, US.
  Fact 0.1499: Willard, Miss Constance was 21 years old.
  Fact 0.1500: Willard, Miss Constance is from Duluth, Minnesota, US.
  Fact 0.1501: Willard, Miss Constance boarded at Southampton.
  Fact 0.1502: Willard, Miss Constance's destination is Duluth, Minnesota, US.
  Fact 0.1503: Willard, Miss Constance has 8 in the lifeboat or body column.
  Fact 0.1504: Williams, Mr. Charles Duane was 51 years old.
  Fact 0.1505: Williams, Mr. Charles Duane is from Geneva, Switzerland.
  Fact 0.1506: Williams, Mr. Charles Duane boarded at Cherbourg.
  Fact 0.1507: Williams, Mr. Charles Duane's destination is Radnor, Pennsylvania, US.
  Fact 0.1508: Williams, Mr. Richard Norris II was 21 years old.
  Fact 0.1509: Williams, Mr. Richard Norris II is from Geneva, Switzerland.
  Fact 0.1510: Williams, Mr. Richard Norris II boarded at Cherbourg.
  Fact 0.1511: Williams, Mr. Richard Norris II's destination is Radnor, Pennsylvania, US.
  Fact 0.1512: Woolner, Mr. Hugh was 45 years old.
  Fact 0.1513: Woolner, Mr. Hugh is from London, England, UK.
  Fact 0.1514: Woolner, Mr. Hugh boarded at Southampton.
  Fact 0.1515: Woolner, Mr. Hugh's destination is New York City.
  Fact 0.1516: Wright, Mr. George was 62 years old.
  Fact 0.1517: Wright, Mr. George is from Halifax, Nova Scotia, Canada.
  Fact 0.1518: Wright, Mr. George boarded at Southampton.
  Fact 0.1519: Wright, Mr. George's destination is Halifax, Nova Scotia, Canada.
  Fact 0.1520: Young, Miss Marie Grice was 36 years old.
  Fact 0.1521: Young, Miss Marie Grice is from New York City.
  Fact 0.1522: Young, Miss Marie Grice boarded at Cherbourg.
  Fact 0.1523: Young, Miss Marie Grice's destination is New York City.
  Fact 0.1524: Young, Miss Marie Grice has 8 in the lifeboat or body column.

Retrieved items:
<item 1>
Passengers of the RMS Titanic
Frank M. Warren , Sr . George Dennick Wick George Dunton Widener Harry Elkins Widener Duane Williams George Henry Wright Roderick Chisholm Survivors ( last
living ) Rhoda Abbott Trevor Allison Lillian Asplund Madeleine Astor Ruth Becker Lawrence Beesley Karl Behr Dickinson Bishop Mauritz Håkan Björnström -
Steffansson Elsie Bowerman Francis Browne Margaret `` Molly '' Brown Daniel Buckley Alden Caldwell Helen Churchill Candee Charlotte Drake Cardeza Lucile Carter
Gladys Cherry Millvina Dean Margaret Devaney Sir Cosmo Duff - Gordon Lucy , Lady Duff - Gordon Ethel Flora Fortune Dorothy Gibson Archibald Gracie IV Frank John
William Goldsmith Edith Haisman Henry S. Harper Eva Hart Margaret Bechstein Hays Masabumi Hosono J. Bruce Ismay Eleanor Ileen Johnson Louise Laroche Louise Kink
Margaret Mannion Michel Marcel Navratil Alfred Nourney Arthur Godfrey Peuchen Winnifred Quick Marjorie Newell Robb Edith Rosenbaum Noël Leslie ,
</item 1>

<item 2>
Passengers of the RMS Titanic
Of the 711 passengers and crew rescued by the Carpathia , six , including first class passenger William F. Hoyt , either died in a lifeboat during the night or
on board the Carpathia the next morning , and were buried at sea . In the days following the sinking , several ships sailed to the disaster area to recover
victims ' bodies . The White Star Line chartered the cable ship Mackay - Bennett from Halifax , Nova Scotia to retrieve bodies . Three other ships followed in
the search : the cable ship Minia , the lighthouse supply ship Montmagny and the sealing vessel Algerine . Each ship left with embalming supplies , undertakers
, and clergy . Upon recovery , each body retrieved by the Mackay - Bennett was numbered and given as detailed a description as possible to help aid in
identification . The physical appearance of each body -- height , weight , age , hair and eye colour , visible birthmarks , scars or tattoos , was catalogued
and any personal effects on the bodies were gathered and placed in small canvas bags corresponding to their number .
</item 2>

<item 3>
Passengers of the RMS Titanic
In the book Titanic : Men and Children First ( October 1998 ) , Judith Geller stated that `` officially were 154 Syrians on board the Titanic and 29 were saved
: four men , five children and 20 women '' . Chinese passengers ( edit ) Eight passengers are listed on the passenger list with a home country of China , from
the hometown of Hong Kong , with a destination of New York . Six of these persons survived the disaster . They were not permitted to enter the United States due
to the Chinese Exclusion Act . Survivors and victims ( edit ) Main articles : Sinking of the RMS Titanic and Lifeboats of the RMS Titanic On the night of 14
April 1912 at around 11 : 40 pm , while the RMS Titanic was sailing about 400 miles ( 640 km ) south of the Grand Banks of Newfoundland , the ship struck an
iceberg and began to sink . Shortly before midnight , Captain Edward Smith ordered the ship 's lifeboats to be readied and a distress call was sent out .
</item 3>

<item 4>
Passengers of the RMS Titanic
Victims whose remains were recovered after the sinking are listed with a superscript next to the body number , indicating the recovery vessel : MB -- CS Mackay
- Bennett ( bodies 1 -- 306 ) M -- CS Minia ( bodies 307 -- 323 ) MM -- CGS Montmagny ( bodies 326 -- 329 ) A -- SS Algerine ( body 330 ) O -- RMS Oceanic (
bodies 331 -- 333 ) I -- SS Ilford ( body 334 ) OT -- SS Ottawa ( body 335 ) Numbers 324 and 325 were unused , and the six bodies buried at sea by the Carpathia
also went unnumbered . First class ( edit ) Name Age Hometown Boarded Destination Lifeboat Body Allen , Miss Elizabeth Walton 29 St. Louis , Missouri , US
Southampton St. Louis , Missouri , US Allison , Mr. Hudson Joshua Creighton 30 Montreal , Quebec , Canada Southampton Montreal , Quebec , Canada 135 and
chauffeur , Mr.
</item 4>

<item 5>
Passengers of the RMS Titanic
( Halifax ) Related Ships RMS Olympic RMS Carpathia HMHS Britannic CS Mackay - Bennett SS Californian Replica Titanic Titanic II Others White Star Line Titanic
Historical Society Encyclopedia Titanica Halomonas titanicae Women and children first SOS Robert Ballard Last remaining survivors of the RMS Titanic disaster
2000 -- 2009 Millvina Dean ( 1912 -- 2009 ) Barbara West ( 1911 -- 2007 ) Lillian Asplund ( 1906 -- 2006 ) Winnifred Quick ( 1904 -- 2002 ) Michel Navratil (
1908 -- 2001 ) 1995 -- 1999 Eleanor Johnson ( 1910 -- 1998 ) Louise Laroche ( 1910 -- 1998 ) Edith Brown ( 1896 -- 1997 ) Eva Hart ( 1905 -- 1996 ) Beatrice
Sandström ( 1910 -- 1995 ) 1990 -- 1994 Robertha `` Bertha '' Watt ( 1899 -- 1993 ) Ellen Shine ( 1891 -- 1993 ) Marjorie Newell ( 1889 -- 1992 ) Louise
</item 5>

<item 6>
Passengers of the RMS Titanic
ISBN 978 - 0 - 14 - 004757 - 8 . Jump up ^ `` Passenger List and Survivors of Steamship Titanic '' . United States Senate Inquiry. 30 July 1912 . Retrieved 15
July 2011 . Jump up ^ Hall , Wayne ( 1986 ) . `` Social Class and Survival on the SS Titanic '' ( PDF ) . Social Science & Medicine . 22 ( 6 ) : 687 -- 690 .
Jump up ^ Barratt , Nick ( 2009 ) . Lost Voices From the Titanic : The Definitive Oral History . London : Random House . p. 93 . ISBN 978 - 1 - 84809 - 151 - 1
. Jump up ^ Howells , Richard ( 1999 ) . The Myth of the Titanic . United Kingdom : MacMillan Press . p. 18 . ISBN 978 - 0 - 333 - 72597 - 9 .
</item 6>

<item 7>
Passengers of the RMS Titanic
Although several people managed to reach this lifeboat , three died during the night . When Fifth Officer Harold Lowe and six crewmen returned to the wreck site
after the sinking with an empty lifeboat to pick up survivors , they rescued surviving passengers from Collapsible A , but left the three dead bodies in the
boat : Thomson Beattie , a first - class passenger , and two crew members , a fireman and a seaman . After their retrieval from Collapsible A by Oceanic , the
bodies were buried at sea . Passenger list ( edit ) Breakdown of casualties according to the British Board of Trade report ( click for detail ) The following is
a full list of known passengers who sailed on the maiden voyage of the RMS Titanic . Included in this list are the nine - member Guarantee Group and the eight
members of the ship 's band , listed as both passengers and crew . They are also included in the list of crew members on board RMS Titanic . Passengers are
colour - coded , indicating whether they were saved or perished . The passenger did not survive The passenger survived Survivors are listed with the lifeboat
from which they were known to be rescued .
</item 7>

<item 8>
Passengers of the RMS Titanic
1 British inquiry US inquiry Wreck of Titanic Maritime Memorial Act Deck officers Edward J. Smith ( Captain ) Henry Tingle Wilde ( Chief Officer ) William
McMaster Murdoch ( First Officer ) Charles H. Lightoller ( Second Officer ) Herbert Pitman ( Third Officer ) Joseph G. Boxhall ( Fourth Officer ) Harold G. Lowe
( Fifth Officer ) James Paul Moody ( Sixth Officer ) Joseph Bell ( Machine Room Manager ) Crew members Frederick Barrett Harold Bride William Denton Cox Sid
Daniels Alfred Frank Evans Frederick Fleet Luigi Gatti Robert Hichens Violet Jessop Charles Joughin Reginald Lee Evelyn Marsden William Mintram Jack Phillips
George Symons Passengers Fatalities Allison family Thomas Andrews John Jacob Astor IV David John Bowen Archibald Butt Thomas Byles Walter Donald Douglas Edith
Corse Evans Annie Funk Jacques Futrelle Sidney Leslie Goodwin Benjamin Guggenheim John Harper Wallace Hartley Charles Melville Hays Edward Austin Kent Joseph
Philippe Lemercier Laroche Francis Davis Millet Harry Markland Molson Michel Navratil Eino Viljami Panula W.T. Stead Ida Straus Isidor Straus John Borland
Thayer Jr . Frank M. Warren , Sr .
</item 8>

<item 9>
Passengers of the RMS Titanic
US 201 Mernagh , Mr. John Robert 26 Ballywilliam , Wexford Ireland Queenstown Chicago , Illinois , US Midtsjø , Mr. Karl Albert 21 Krøkstad , Ski Norway
Southampton Chicago , Illinois , US 15 Mihoff , Mr. Stoytcho 28 Gumostnik , Lovec Bulgaria Southampton Chicago , Illinois , US Miles , Mr. Frank 23 Greenwich ,
London England Southampton New York City Mineff , Mr. Ivan 24 Unknown Bulgaria Southampton Coon Rapids , Iowa , US Minkoff , Mr. Lazar 21 Gumostnik , Lovec
Bulgaria Southampton Chicago , Illinois , US Mitkoff , Mr. Mito 23 Unknown Bulgaria Southampton Chicago , Illinois , US Mockler , Miss Ellen Mary 23 Currafarry
, Galway Ireland Queenstown New York City 16 Moen , Mr. Sigurd Hansen 25 Bergen Norway Southampton Minneapolis , Minnesota , US 309 Moor , Mrs. Beila 29 Unknown
Russia Southampton Chicago , Illinois , US 14 Moor , Master Meier 7 Unknown Russia Southampton Chicago , Illinois , US 14 Moore , Mr.
</item 9>

<item 10>
Passengers of the RMS Titanic
Edward Rowland 21 Swanage England Southampton Cleveland , Ohio , US Storey , Mr. Thomas 51 Liverpool , Merseyside England , Southampton New York City 261
Stoytcheff , Mr. Ilia 19 Gumostnik , Lovec Bulgaria Southampton Chicago , Illinois , US Strandberg , Miss Ida Sofia 22 Turku Finland Southampton New York City
Strandén , Mr. Juho Niilonpoika 31 Muljula , Kitee Finland Southampton Duluth , Minnesota , US 9 Strilić , Mr. Ivan 27 Siroka Kula Široka Kula Croatia
Southampton Chicago , Illinois , US Ström , Mrs. Elna Matilda ( née Persson ) 29 Indiana Harbor , Indiana US Southampton Indiana Harbor , Indiana , US Ström ,
Miss Thelma Matilda Wilhelmina Indiana Harbor , Indiana US Southampton Indiana Harbor , Indiana , US Sunderland , Mr. Victor Francis 20 London England
Southampton Cleveland , Ohio , US Sundman , Mr. Johan Julian 44 Munsala Finland Southampton Cheyenne , Wyoming , US 15 Sutehall Jr. , Mr.
</item 10>
=== END ITEM id=7198e6607d695c79 ===

=== ITEM id=415224def888356f ===
Gold sentences and facts:
Sentence 0: Duncan 's diagnosis was publicly confirmed during a CDC news conference the same day . That evening , Duncan reported feeling better and requested to watch a movie . The following morning , Duncan was breathing rapidly and complaining of `` pain all over '' . By the afternoon , however , he was able to eat , and the doctor noted that he was feeling better . The next day , October 3 , Duncan again reported feeling abdominal pain . That evening , the hospital contacted Chimerix , a biotechnology company developing Brincidofovir to combat the disease . The next day , Duncan 's organs were failing , and he was intubated to help him breathe . In the afternoon , the hospital began administering Brincidofovir . Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock . On October 7 , the hospital reported that Duncan 's condition was improving . However , Duncan died at 7 : 51 a.m. on October 8 , becoming the first person to die in the United States of Ebola virus disease and the index patient for the later infections of nurses Pham and Vinson .
  Fact 0.0: The CDC confirmed Duncan's Ebola diagnosis at a news conference on the same day.
  Fact 0.1: That evening, Duncan reported feeling better and requested to watch a movie.
  Fact 0.2: The following morning, Duncan was breathing rapidly and complaining of 'pain all over'.
  Fact 0.3: By the afternoon, Duncan was able to eat.
  Fact 0.4: By the afternoon, the doctor noted that Duncan was feeling better.
  Fact 0.5: On October 3, Duncan again reported feeling abdominal pain.
  Fact 0.6: That evening, October 3, the hospital contacted Chimerix.
  Fact 0.7: Chimerix is a biotechnology company developing Brincidofovir to combat the disease.
  Fact 0.8: On October 4, Duncan's organs were failing.
  Fact 0.9: On October 4, Duncan was intubated to help him breathe.
  Fact 0.10: On October 4 in the afternoon, the hospital began administering Brincidofovir to Duncan.
  Fact 0.11: Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock.
  Fact 0.12: On October 7, the hospital reported that Duncan's condition was improving.
  Fact 0.13: Duncan died at 7:51 a.m. on October 8.
  Fact 0.14: Duncan became the first person to die in the United States of Ebola virus disease.
  Fact 0.15: Duncan was the index patient for the later infections of nurses Pham and Vinson.
Sentence 1: In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola '' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first was reported in September 2014 . Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered .
  Fact 1.0: In December 2014, Ebola virus cases occurred in the United States due to four laboratory-confirmed cases of Ebola virus disease in the United States.
  Fact 1.1: Ebola virus disease is commonly known as Ebola.
  Fact 1.2: Eleven Ebola cases have been reported.
  Fact 1.3: The eleven reported cases include the four cases diagnosed in the United States and seven cases medically evacuated from other countries.
  Fact 1.4: The first of the eleven cases was reported in September 2014.
  Fact 1.5: Nine of the eleven cases contracted the disease outside the US and traveled into the country.
  Fact 1.6: The nine people traveled into the country either as regular airline passengers or as medical evacuees.
  Fact 1.7: Two of those nine people died.
  Fact 1.8: Two people have contracted Ebola in the United States.
  Fact 1.9: Both people who contracted Ebola in the United States were nurses who treated an Ebola patient.
  Fact 1.10: Both nurses have recovered.
Sentence 2: On September 30 , 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 , Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were :
  Fact 2.0: The CDC announced on September 30, 2014 that Thomas Eric Duncan had been diagnosed with Ebola in Dallas, Texas.
  Fact 2.1: Thomas Eric Duncan was reportedly 42 years old.
  Fact 2.2: CDC reports later corrected Thomas Eric Duncan's age to 45 years old.
  Fact 2.3: Thomas Eric Duncan was a Liberian national.
  Fact 2.4: Thomas Eric Duncan was visiting the United States from Liberia.
  Fact 2.5: Thomas Eric Duncan had been visiting family in Dallas.
  Fact 2.6: Thomas Eric Duncan was treated at Texas Health Presbyterian Hospital Dallas.
  Fact 2.7: By October 4, Thomas Eric Duncan's condition had deteriorated from 'serious but stable' to 'critical'.
  Fact 2.8: Thomas Eric Duncan died of Ebola on October 8.
  Fact 2.9: Three other cases were diagnosed in the United States as of October 2014, and they are listed after this sentence.
Sentence 3: Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
  Fact 3.0: The page is about Ebola virus disease in the U.S.
  Fact 3.1: The page includes a map of Ebola cases and infrastructure throughout the U.S.
  Fact 3.2: The page has categories for cases contracted in the U.S., cases first diagnosed in the U.S., and cases evacuated to the U.S. from other countries.
  Fact 3.3: The table lists 7 cases evacuated to the U.S. from other countries.
  Fact 3.4: The table lists 11 total cases.
  Fact 3.5: The table has a 'Deaths' label.
  Fact 3.6: The table lists 9 recoveries from Ebola.
  Fact 3.7: The table lists 0 active cases.

Retrieved items:
<item 1>
Ebola virus cases in the United States
Retrieved October 7 , 2014 . Jump up ^ Sifferlin , Alexandra ( October 7 , 2014 ) . `` Dallas Ebola Patient Improves Slightly '' . TIME.com . Retrieved October
8 , 2014 . Jump up ^ `` Texas Ebola Patient Thomas Eric Duncan Has Died '' . ABC News . Retrieved October 8 , 2014 . Jump up ^ `` Cases of Ebola Diagnosed in
the United States '' . CDCP . Retrieved October 16 , 2014 . Jump up ^ `` Found : Homeless Man Sought in Ebola Case Being Monitored '' . NBC News . October 5 ,
2014 . Retrieved October 5 , 2014 . Jump up ^ `` 10 people at ' high risk ' for Ebola as 50 people monitored daily after contact with Dallas Ebola patient '' .
New York Daily News . October 3 , 2014 . Retrieved October 3 , 2014 . Jump up ^ `` Hospital : Dallas Ebola patient in critical condition '' . CNN . October 4 ,
2014 .
</item 1>

<item 2>
Ebola virus cases in the United States
Medical evacuations from West Africa ( edit ) Local transmissions -- no deaths : Texas Initial cases -- deaths : Texas Initial cases -- no deaths : New York
Medically evacuated cases -- no deaths : Georgia , Maryland Medically evacuated cases -- deaths : Nebraska By the time the Ebola virus epidemic in West Africa
was over in June 2016 , seven Americans had been evacuated to the U.S. for treatment after contracting Ebola virus while working with medical teams trying to
stop the epidemic . Kent Brantly , a physician and medical director in Liberia for the aid group Samaritan 's Purse , and co-worker Nancy Writebol were infected
in July 2014 , while working in Monrovia . Both were flown to the United States at the beginning of August via the CDC 's Aeromedical Biological Containment
System for further treatment in Atlanta 's Emory University Hospital . On August 21 , Brantly and Writebol recovered and were discharged . Brantly has since
donated blood to three others with Ebola ( Sacra , Mukpo and Pham ) .
</item 2>

<item 3>
Ebola virus cases in the United States
WYFF4 . Retrieved October 3 , 2014 . Jump up ^ `` First US Ebola case Thomas Duncan ' critical ' '' . BBC News . October 4 , 2014 . Retrieved October 4 , 2014 .
Jump up ^ `` Texas Ebola patient dies '' . CNN . October 8 , 2014 . Retrieved October 8 , 2014 . ^ Jump up to : Fernandez , Manny ( October 12 , 2014 ) . ``
Texas Health Worker Tests Positive for Ebola '' . New York Times . Retrieved October 12 , 2014 . Jump up ^ `` Ebola outbreak : Second Texas health worker '
tests positive ' '' . BBC News . October 15 , 2014 . Retrieved October 15 , 2014 . ^ Jump up to : `` New York Doctor Just Back From Africa Has Ebola '' . NBC
News . October 23 , 2014 . Retrieved October 23 , 2014 . Jump up ^ Dan Diamond ( October 20 , 2014 ) .
</item 3>

<item 4>
Ebola virus cases in the United States
Ebola virus cases in the United States - wikipedia Ebola virus cases in the United States Jump to : navigation , search This article is about Ebola virus cases
in the United States starting in 2014 . For the 2013 - 2016 outbreak in Africa , see Ebola virus epidemic in West Africa . For the U.S. discovery of a different
species of the Ebolavirus genus in 1989 , see Reston virus . Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases
contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola
'' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first
was reported in September 2014 .
</item 4>

<item 5>
Ebola virus cases in the United States
ABC News . September 25 , 2014 . Retrieved September 25 , 2014 . Jump up ^ Grady , Denise ( December 7 , 2014 ) . `` An Ebola Doctor 's Return From the Edge of
Death '' . New York Times . Retrieved December 8 , 2014 . Jump up ^ du Lac , J. Freedom ( October 15 , 2014 ) . `` Little mentioned patient at Emory expects to
be discharged very soon free from the Ebola virus '' . Washington Post . Jump up ^ `` Ebola patient arrives at Atlanta hospital '' . WCVB Boston . September 9 ,
2014 . Retrieved September 10 , 2014 . Jump up ^ Addison , Stephen ( September 18 , 2014 ) . `` British Ebola survivor flies to United States for blood donation
'' . Reuters . Retrieved September 25 , 2014 . Jump up ^ `` 2014 Ebola Outbreak in West Africa - September 25 , 2014 '' . CDC . September 25 , 2014 .
</item 5>

<item 6>
Ebola virus cases in the United States
Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those
nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered . On September 30
, 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national
visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at
Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 ,
Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were : October 11 , 2014 , a nurse , Nina Pham , who had provided
care to Duncan at the hospital . October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan .
</item 6>

<item 7>
Ebola virus cases in the United States
Jump up ^ `` First diagnosed case of Ebola in the U.S. '' CNN . September 30 , 2014 . Retrieved October 27 , 2014 . Jump up ^ `` New Details About Ebola In
Dallas Revealed In CDC Report '' . HuffingtonPost . November 14 , 2014 . Retrieved November 14 , 2014 . Jump up ^ CDC confirms first ever Ebola case in United
States , cdc.gov ; accessed October 9 , 2014 . Jump up ^ Fernandez , Manny ; Philipps , Dave ( October 9 , 2014 ) . `` Death of Thomas Eric Duncan in Dallas
Fuels Alarm Over Ebola '' . The New York Times . Retrieved October 9 , 2014 . ^ Jump up to : Fernandez , Manny ; Onishi , Norimitsu ( October 1 , 2014 ) . ``
U.S. Patient Aided Ebola Victim in Liberia '' . The New York Times . ^ Jump up to : `` Thomas Eric Duncan : From healthy to Ebola '' . WYFF4 . Retrieved October
3 , 2014 .
</item 7>

<item 8>
Ebola virus cases in the United States
See also ( edit ) Ebola virus epidemic in Guinea Ebola virus epidemic in Liberia Ebola virus epidemic in Nigeria Ebola virus epidemic in Sierra Leone Ebola
virus disease in Spain List of Ebola outbreaks West Nile virus in the United States Notes ( edit ) Jump up ^ Duncan contracted Ebola in Liberia and was
diagnosed in the U.S. Two of the victims had had direct contact with Duncan . One victim was diagnosed in New York City ; his case is unrelated to the Texas
cases . ^ Jump up to : This takes into account the seven Americans who were medically evacuated from West Africa with Ebola . References ( edit ) Jump up ^ ``
Who are the American Ebola patients ? '' , CNN , October 6 , 2014 Jump up ^ `` Nebraska Hospital : Surgeon With Ebola Has Died '' . ABC News . November 17 ,
2014 . Retrieved November 17 , 2014 . Jump up ^ `` 2014 Ebola Outbreak - Case Count '' . CDC . December 15 , 2014 . Retrieved December 16 , 2014 . Jump up ^ ``
First diagnosed case of Ebola in the U.S. '' CNN .
</item 8>

<item 9>
Ebola virus cases in the United States
In addition , on September 21 , a CDC employee , who showed no symptoms of the disease , was flown back to the United States as a preventative measure after low
- risk exposure with a healthcare worker ; he posed no risks to his family or the United States , and he is not known to have died . On September 28 , a fourth
American doctor was admitted to National Institutes of Health hospital . On October 2 , NBC News photojournalist Ashoka Mukpo , covering the outbreak in Liberia
, tested positive for Ebola after showing symptoms . Four other members of the NBC team , including physician Nancy Snyderman , were being closely monitored for
symptoms . Mukpo was evacuated on October 6 to the University of Nebraska Medical Center for treatment in their isolation unit . On October 21 , Mukpo was
declared Ebola - free and allowed to return to his home in Rhode Island . A government official in Sierra Leone announced on November 13 , that a doctor from
Sierra Leone , a permanent resident of the United States married to a U.S. citizen , would be transported to the Nebraska Medical Center for treatment for Ebola
. The doctor , identified later as Dr.
</item 9>

<item 10>
Ebola virus cases in the United States
October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan . October 23 , 2014 , physician Craig Spencer , diagnosed in New York City ; he had just
returned from working with Doctors Without Borders in Guinea , a country in west Africa . Hundreds of people were tested or monitored for potential Ebola virus
infection , but the two nurses were the only confirmed cases of locally transmitted Ebola . Public health experts and the Obama administration opposed
instituting a travel ban on Ebola endemic areas , stating that it would be ineffective and would paradoxically worsen the situation . No one both contracted and
died of Ebola virus disease while in the United States . No new cases have been diagnosed in the United States since Craig Spencer was released from a New York
City hospital on November 11 , 2014 . Contents ( hide ) 1 Cases diagnosed in U.S. 1.1 First case : Thomas Eric Duncan 1.1. 1 Thomas Eric Duncan in Liberia 1.1.
2 Duncan 's illness in Dallas 1.1. 3 Contact tracing 1.1.
</item 10>
=== END ITEM id=415224def888356f ===

=== ITEM id=e805a9661bf1b476 ===
Gold sentences and facts:
Sentence 0: Barbara Eden as Stella Johnson , a widowed cosmetics salesperson with a young daughter , whose joie de vivre , her principles , and her individuality anger most of the powerful people in town and on the Parent Teachers Association board . She inevitably becomes PTA president after embarrassing Flora , the snobby former President and her equally snobby friends . Ronny Cox as Willis `` Will '' Newton , a wealthy member of the PTA who does n't agree with Flora and her snobby allies and is in love with Stella . He and Stella get married at the end of the movie . Nanette Fabray as Alice Finley , Stella 's best friend and co-conspirator against the snobs of Harper Valley . She owns and operates the La Moderne Beauty Shop . Louis Nye as Kirby Baker , a local real estate agent who is allied with Flora and her friends . Notorious for being a lecherous playboy and infamous for impregnating his secretaries , one of whom , Gladys Wilmot , left town to see an abortionist in Cincinnati . As part of a plan to discredit Stella , he tries to foreclose on Stella 's house , which infuriates her . He then gets arrested and jailed for attacking Myrna Wong , a friend of Stella 's , who rebuffs his advances , throwing him by use of martial arts and destroying the office . John Fiedler as Bobby Taylor , a man on the PTA who lusts after Stella . Overly dominated by his wife , Holly . Although open to allow Stella to speak , he is not above having a fellow PTA board member kidnapped to commit election fraud . He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others ) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main antagonist . She is the pompous and extremely snobbish president of the PTA board who thoroughly despises Stella ( who defeats her for president at the end of the movie ) and all she stands for . So desperate is she to retain her power and prestige in the community that she has a fellow member of the PTA ( Skeeter Duggan ) kidnapped so she and her allies can commit election fraud . She is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon . Molly Dodd as Olive Glover , the recording secretary of the PTA who is a gambling addict . Clearly the most corrupt and most vindictive member of the PTA board , she is eventually arrested for embezzlement after having a load of hay and horse manure dumped on top of her . She is the one who initially angers Stella by sending her the letter denouncing her , and tries to get Dee 's friend , Mavis , arrested for embezzlement . Fay DeWitt as Willamae Jones , a snide biology teacher at the junior high school who is also a member of the PTA . She makes no secret of the fact that she favors the Reilly twins , due to the connections between her and Flora . Notorious for being a nymphomaniac ( though she hides it under a prim - and - proper façade ) , she is caught in flagrante delicto with Barney Crunk , the town 's mailman , on film . DeVera Marcus as Holly Taylor , Bobby 's obviously overbearing wife , who tends to drink a lot when Bobby is out of town , and does n't approve of Bobby 's lusting , or of anything else he does ( this is different from the character in the song , who `` seems to use a lot of ice whenever ( her husband ) 's away '' , implying adultery with the iceman ) . However , despite her ties to Flora and her scheming friends , she , like Shirley Thompson , is spared any retribution from Stella , due to Stella refusing to date Bobby . Louise Foley as Mavis , Dee Johnson 's best friend , who is nearly framed for a crime she did not commit by the vindictive Olive Glover . Brian Cook as Carlyle , a popular track star who chooses Dee , after she gets a makeover , and angers Bettina Reilly in the process . Laura Teige as Bettina Reilly , One of Flora 's twin granddaughters , who wants to be with Carlyle , and is jealous when he chooses Dee over her . She tends to be as snotty as her grandmother . Jan Teige as Edwina Reilly , Bettina 's twin sister , who is just as snotty as her . Irene Yah - Ling Sun as Myrna Wong , a friend of Stella 's who is a judo expert ; she is assaulted by Kirby Baker ( after leading him on ; she destroys him and his office in retaliation ) and has him sent to jail for assault . Tobias Anderson as `` Cranky '' Barney Crunk , the town 's nosy mailman who had the bad habit of peeking in everyone 's mail . He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson - Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her house is destroyed by pink - painted elephants .
  Fact 0.0: Barbara Eden plays Stella Johnson.
  Fact 0.1: Stella Johnson is a widowed cosmetics salesperson.
  Fact 0.2: Stella Johnson has a young daughter.
  Fact 0.3: Stella Johnson's joie de vivre angers most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.4: Stella Johnson's principles anger most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.5: Stella Johnson's individuality angers most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.6: Stella Johnson inevitably becomes PTA president after embarrassing Flora, the snobby former President.
  Fact 0.7: Stella Johnson inevitably becomes PTA president after embarrassing Flora's equally snobby friends.
  Fact 0.8: Ronny Cox plays Willis 'Will' Newton.
  Fact 0.9: Will Newton is a wealthy member of the PTA.
  Fact 0.10: Will Newton does not agree with Flora.
  Fact 0.11: Will Newton does not agree with Flora's snobby allies.
  Fact 0.12: Will Newton is in love with Stella.
  Fact 0.13: Will Newton and Stella get married at the end of the movie.
  Fact 0.14: Nanette Fabray plays Alice Finley.
  Fact 0.15: Alice Finley is Stella's best friend.
  Fact 0.16: Alice Finley is Stella's co-conspirator against the snobs of Harper Valley.
  Fact 0.17: Alice Finley owns the La Moderne Beauty Shop.
  Fact 0.18: Alice Finley operates the La Moderne Beauty Shop.
  Fact 0.19: Louis Nye plays Kirby Baker.
  Fact 0.20: Kirby Baker is a local real estate agent.
  Fact 0.21: Kirby Baker is allied with Flora and her friends.
  Fact 0.22: Kirby Baker is notorious for being a lecherous playboy.
  Fact 0.23: Kirby Baker is infamous for impregnating his secretaries.
  Fact 0.24: Gladys Wilmot is one of Kirby Baker's secretaries.
  Fact 0.25: Kirby Baker impregnated Gladys Wilmot.
  Fact 0.26: Gladys Wilmot left town to see an abortionist in Cincinnati.
  Fact 0.27: As part of a plan to discredit Stella, Kirby Baker tries to foreclose on Stella's house.
  Fact 0.28: Stella is infuriated by Kirby Baker's attempt to foreclose on her house.
  Fact 0.29: Kirby Baker gets arrested and jailed for attacking Myrna Wong.
  Fact 0.30: Myrna Wong is a friend of Stella's.
  Fact 0.31: Myrna Wong rebuffs Kirby Baker's advances.
  Fact 0.32: Myrna Wong throws Kirby Baker by use of martial arts.
  Fact 0.33: Myrna Wong destroys Kirby Baker's office.
  Fact 0.34: John Fiedler plays Bobby Taylor.
  Fact 0.35: Bobby Taylor is a man on the PTA.
  Fact 0.36: Bobby Taylor lusts after Stella.
  Fact 0.37: Bobby Taylor is overly dominated by his wife, Holly.
  Fact 0.38: Bobby Taylor is open to allowing Stella to speak.
  Fact 0.39: Bobby Taylor is not above having a fellow PTA board member kidnapped to commit election fraud.
  Fact 0.40: Bobby Taylor was arrested for indecent exposure.
  Fact 0.41: Bobby Taylor was set up by Stella.
  Fact 0.42: Ron Masak plays Herbie Maddox.
  Fact 0.43: Herbie Maddox is the head bartender at Kelly's bar.
  Fact 0.44: Herbie Maddox is Stella's ally against Flora and her friends.
  Fact 0.45: Herbie Maddox has a crush on Stella.
  Fact 0.46: Herbie Maddox is one of Stella's friends.
  Fact 0.47: Clint Howard plays Corley.
  Fact 0.48: Corley is a young delivery person.
  Fact 0.49: Corley works for the local florist shop.
  Fact 0.50: Corley brings Stella flowers from Will.
  Fact 0.51: Susan Swift plays Dee Johnson.
  Fact 0.52: Dee Johnson is Stella's 14-year-old daughter.
  Fact 0.53: Dee Johnson is threatened with expulsion from school if her mother does not conform to the way the PTA wants her to behave.
  Fact 0.54: Dee Johnson has a hand in assisting her mother and Alice with one of their revenge schemes.
  Fact 0.55: Dee Johnson loses her braces.
  Fact 0.56: Dee Johnson gets a makeover.
  Fact 0.57: Dee Johnson ends up with a boyfriend, Carlyle.
  Fact 0.58: Pat Paulsen plays Otis Harper.
  Fact 0.59: Otis Harper is the descendant of the founding family of Harper Valley.
  Fact 0.60: Otis Harper is a PTA member allied with Flora and her friends.
  Fact 0.61: Otis Harper has a tendency to drink all the time.
  Fact 0.62: Otis Harper's drinking annoys his wife, Bertha.
  Fact 0.63: Otis Harper had a stampede of pink-painted elephants storm through his house.
  Fact 0.64: The stampede of pink-painted elephants destroyed Otis Harper's house.
  Fact 0.65: Amzie Strickland plays Shirley Thompson.
  Fact 0.66: Shirley Thompson is a member of the PTA board.
  Fact 0.67: Shirley Thompson tends to drink a lot of gin.
  Fact 0.68: Shirley Thompson is allied with Flora and her friends.
  Fact 0.69: Shirley Thompson is one of four PTA members spared any retribution from Stella.
  Fact 0.70: The other three PTA members spared any retribution from Stella are Will, Skeeter and Holly Taylor.
  Fact 0.71: Shirley Thompson was seen as the most sympathetic member of the board.
  Fact 0.72: Shirley Thompson tended to be more manipulated than manipulative.
  Fact 0.73: It was never revealed whether Shirley Thompson retained her seat on the PTA.
  Fact 0.74: It is presumed that Shirley Thompson became more sympathetic towards Stella.
  Fact 0.75: Shirley Thompson was seen waving goodbye at Olive as Olive was being arrested.
  Fact 0.76: Bob Hastings plays Skeeter Duggan.
  Fact 0.77: Skeeter Duggan is a member of the PTA board.
  Fact 0.78: Skeeter Duggan is the town's Notary public.
  Fact 0.79: Skeeter Duggan is sympathetic to Stella.
  Fact 0.80: Dutch and Tex kidnap Skeeter Duggan at Flora Simpson-Reilly's orders.
  Fact 0.81: Audrey Christie plays Flora Simpson-Reilly.
  Fact 0.82: Flora Simpson-Reilly is the movie's main antagonist.
  Fact 0.83: Flora Simpson-Reilly is the pompous and extremely snobbish president of the PTA board.
  Fact 0.84: Flora Simpson-Reilly thoroughly despises Stella.
  Fact 0.85: Stella defeats Flora Simpson-Reilly for president at the end of the movie.
  Fact 0.86: Flora Simpson-Reilly despises all that Stella stands for.
  Fact 0.87: Flora Simpson-Reilly is desperate to retain her power and prestige in the community.
  Fact 0.88: Flora Simpson-Reilly has a fellow PTA member, Skeeter Duggan, kidnapped so she and her allies can commit election fraud.
  Fact 0.89: Flora Simpson-Reilly is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon.
  Fact 0.90: Molly Dodd plays Olive Glover.
  Fact 0.91: Olive Glover is the recording secretary of the PTA.
  Fact 0.92: Olive Glover is a gambling addict.
  Fact 0.93: Olive Glover is clearly the most corrupt and most vindictive member of the PTA board.
  Fact 0.94: Olive Glover is eventually arrested for embezzlement after having a load of hay and horse manure dumped on top of her.
  Fact 0.95: Olive Glover initially angers Stella by sending her the letter denouncing her.
  Fact 0.96: Olive Glover tries to get Dee's friend, Mavis, arrested for embezzlement.
  Fact 0.97: Fay DeWitt plays Willamae Jones.
  Fact 0.98: Willamae Jones is a snide biology teacher at the junior high school.
  Fact 0.99: Willamae Jones is a member of the PTA.
  Fact 0.100: Willamae Jones favors the Reilly twins.
  Fact 0.101: Willamae Jones favors the Reilly twins due to the connections between her and Flora.
  Fact 0.102: Willamae Jones is notorious for being a nymphomaniac.
  Fact 0.103: Willamae Jones hides being a nymphomaniac under a prim-and-proper facade.
  Fact 0.104: Willamae Jones is caught in flagrante delicto with Barney Crunk, the town's mailman, on film.
  Fact 0.105: DeVera Marcus plays Holly Taylor.
  Fact 0.106: Holly Taylor is Bobby's obviously overbearing wife.
  Fact 0.107: Holly Taylor tends to drink a lot when Bobby is out of town.
  Fact 0.108: Holly Taylor does not approve of Bobby's lusting.
  Fact 0.109: Holly Taylor does not approve of anything else Bobby does.
  Fact 0.110: The Holly Taylor character differs from the character in the song, who seems to use a lot of ice whenever her husband is away, implying adultery with the iceman.
  Fact 0.111: Holly Taylor has ties to Flora and her scheming friends.
  Fact 0.112: Holly Taylor, like Shirley Thompson, is spared any retribution from Stella.
  Fact 0.113: Stella spares Holly Taylor due to Stella refusing to date Bobby.
  Fact 0.114: Louise Foley plays Mavis.
  Fact 0.115: Mavis is Dee Johnson's best friend.
  Fact 0.116: Mavis is nearly framed for a crime she did not commit by the vindictive Olive Glover.
  Fact 0.117: Brian Cook plays Carlyle.
  Fact 0.118: Carlyle is a popular track star.
  Fact 0.119: Carlyle chooses Dee after she gets a makeover.
  Fact 0.120: Carlyle's choice of Dee angers Bettina Reilly in the process.
  Fact 0.121: Laura Teige plays Bettina Reilly.
  Fact 0.122: Bettina Reilly is one of Flora's twin granddaughters.
  Fact 0.123: Bettina Reilly wants to be with Carlyle.
  Fact 0.124: Bettina Reilly is jealous when Carlyle chooses Dee over her.
  Fact 0.125: Bettina Reilly tends to be as snotty as her grandmother.
  Fact 0.126: Jan Teige plays Edwina Reilly.
  Fact 0.127: Edwina Reilly is Bettina's twin sister.
  Fact 0.128: Edwina Reilly is just as snotty as Bettina.
  Fact 0.129: Irene Yah-Ling Sun plays Myrna Wong.
  Fact 0.130: Myrna Wong is a friend of Stella's.
  Fact 0.131: Myrna Wong is a judo expert.
  Fact 0.132: Myrna Wong is assaulted by Kirby Baker after she leads him on.
  Fact 0.133: Myrna Wong destroys Kirby Baker and his office in retaliation.
  Fact 0.134: Myrna Wong has Kirby Baker sent to jail for assault.
  Fact 0.135: Tobias Anderson plays 'Cranky' Barney Crunk.
  Fact 0.136: Barney Crunk is the town's nosy mailman.
  Fact 0.137: Barney Crunk had the bad habit of peeking in everyone's mail.
  Fact 0.138: Barney Crunk is caught in flagrante delicto on film with Willamae Jones.
  Fact 0.139: Royce D. Applegate plays Dutch.
  Fact 0.140: Dutch is one of two kidnappers who abduct Skeeter Duggan at Flora Simpson-Reilly's orders.
  Fact 0.141: J.J. Barry plays Tex.
  Fact 0.142: Tex is the mastermind of the kidnap plot to get Skeeter Duggan out of the way.
  Fact 0.143: The kidnap plot lets Flora Simpson-Reilly and her friends commit election fraud.
  Fact 0.144: Tex is arrested.
  Fact 0.145: Dutch is arrested.
  Fact 0.146: Tex and Dutch rat out Flora and her friends on their scheme.
  Fact 0.147: Pitt Herbert plays Henry Reilly.
  Fact 0.148: Henry Reilly is Flora's overly dominated husband.
  Fact 0.149: Henry Reilly is the grandfather of Edwina and Bettina.
  Fact 0.150: Henry Reilly is wealthy.
  Fact 0.151: Henry Reilly is very weak-willed.
  Fact 0.152: Henry Reilly is known for acquiescing to his wife's schemes.
  Fact 0.153: Arlen Stuart plays Bertha Harper.
  Fact 0.154: Bertha Harper is the wife of Otis Harper.
  Fact 0.155: Bertha Harper gets annoyed by her husband's continual drinking.
  Fact 0.156: Bertha Harper is terrified when her house is destroyed by pink-painted elephants.

Retrieved items:
<item 1>
Harper Valley PTA (film)
The letter denounces her for her not following the societal mores of the day and the community and further stating that if she does not change her ways more to
the board 's liking , Dee will be punished for her mother 's `` sins '' by being expelled from school . Infuriated by the board 's supposed superiority and
their glaring hypocrisy , Stella storms to the PTA meeting being held that day , and proceeds to tell most of the PTA members off by exposing their hidden
skeletons for the town to see . After her house is TP'ed ( Toilet - Papered ) and a rock with a vile note attached is thrown through her window in retaliation ,
Stella prepares to get even with those who would want her driven out of town . She teams up with her friends , beautician Alice Finley and bartender Herbie
Maddox , and ( along with Dee 's help ) wreaks hilarious and justified revenge on six of the hypocritical PTA members , with methods including : 1 ) tricking a
married male board member who has repeatedly tried to date Stella into a disastrous rendezvous , 2 ) embarrassing Mrs.
</item 1>

<item 2>
Harper Valley PTA (film)
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
Harper Valley PTA (film)
The film starred Barbara Eden , Nanette Fabray , Ronny Cox , Louis Nye and Susan Swift , directed by Richard Bennett , and was mainly released to drive - in
theaters throughout the summer of 1978 . The film has a cult following in fans of the original song . The film 's promotional tagline is : `` The song was
scandalous . The movie is hilarious ! '' Contents ( hide ) 1 Plot 2 Cast and characters 3 Production and release 3.1 Filming 3.2 Release 4 Soundtrack 5
Television series 6 References 7 External links Plot ( edit ) Stella Johnson is a beautiful widowed single mother who lives in the town of Harper Valley , Ohio
. She sells cosmetics door - to - door for the fictitious AngelGlo Cosmetics and is not afraid to enjoy life . Her fourteen - year - old daughter , Dee , is a
student at Harper Valley Junior High School . After leaving school , Dee brings her mother a letter from the school 's Parent Teacher Association board , which
is led by the pompous and snobbish Flora Simpson - Reilly .
</item 3>

<item 4>
Harper Valley PTA (film)
Side 1 `` Harper Valley PTA '' -- performed by Jeannie C. Riley `` Dee 's Visit '' ( instrumental ) -- performed by Nelson Riddle `` Mr. Harper '' -- performed
by Barbara Eden `` Alice 's Place '' ( instrumental ) -- performed by Nelson Riddle `` High School Confidential '' -- performed by Jerry Lee Lewis `` Willie Mae
'' ( instrumental ) -- performed by Nelson Riddle Side 2 `` Harper Valley PTA '' ( instrumental ) -- performed by Nelson Riddle `` Widow Jones '' -- performed
by Barbara Eden `` Twin Tune '' ( instrumental ) -- performed by Nelson Riddle `` Ballad of a Teenage Queen '' -- performed by Johnny Cash `` Ice Cream Disco ''
( instrumental ) -- performed by Nelson Riddle `` Whatever Happened to Charlie Brown '' -- performed by Rita Remington & Carol Channing `` Ending ( Reprise :
Harper Valley PTA ) '' ( instrumental ) -- performed by Nelson Riddle Television series ( edit ) Main article : Harper Valley PTA ( TV series ) In 1981 , Harper
Valley PTA was made into a television sitcom ( created by Sherwood Schwartz ) which aired on NBC from January 1981 to May 1982 .
</item 4>

<item 5>
Harper Valley PTA (film)
He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who
tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others
) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being
manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she
was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is
sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main
antagonist .
</item 5>

<item 6>
Harper Valley PTA (film)
He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson -
Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could
commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry
Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to
his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her
house is destroyed by pink - painted elephants .
</item 6>

<item 7>
Harper Valley PTA (film)
He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally
against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the
local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion
from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge
schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family
of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a
stampede of elephants , painted pink , stormed through his house , destroying it .
</item 7>

<item 8>
Harper Valley PTA (film)
The film grossed over $2 million its opening weekend , $5.8 million in its first three weeks and over $16 million after twelve weeks . The film opened in Los
Angeles on August 2 , 1978 and in New York City on January 12 , 1979 . Soundtrack ( edit ) Harper Valley PTA : Original Soundtrack Recording Soundtrack album by
various artists Released June 1978 Recorded Singleton Sound Studios Nashville , Tennessee Genre Soundtrack Length 29 : 50 Label Plantation Records Producer
Shelby S. Singleton , Jr . Singles from Harper Valley PTA `` Harper Valley PTA ( re-issue ) '' Released : July 1978 In June 1978 , a soundtrack album was
released on vinyl , cassette and 8 - track tape by Plantation Records . To promote the film 's release and its soundtrack , the title song by Jeannie C. Riley
was re-issued as a single . The album made its compact disc premiere through Varèse Sarabande on October 27 , 2017 .
</item 8>

<item 9>
Harper Valley PTA (film)
She is terrified when her house is destroyed by pink - painted elephants . Production and release ( edit ) Filming ( edit ) Harper Valley PTA was filmed in
twenty - seven days from October 3 , 1977 to December 8 , 1977 . It was filmed on location for one week in the town of Lebanon , Ohio and then continued in Los
Angeles , California . The track meet scene was filmed at Simi Valley High School On October 31 , 1977 , production ceased temporarily when Nanette Fabray was
knocked to the ground by a spooked elephant while shooting a scene involving pink elephants in the Hancock Park neighborhood of Los Angeles , California ; she
suffered a severe concussion , bruises and back sprains and was taken to Cedars - Sinai Medical Center where her condition was reported as serious but stable .
Following her recovery , filming resumed on November 30 . Release ( edit ) Harper Valley PTA opened in six theaters in Lebanon , Cincinnati and Dayton , Ohio on
May 23 , 1978 in accordance with executive producer Phil Borack 's plan to release the film first in smaller markets , where regional success could encourage
exhibitors in bigger cities to book the film .
</item 9>

<item 10>
Harper Valley PTA (film)
Harper Valley PTA ( film ) - wikipedia Harper Valley PTA ( film ) Jump to : navigation , search Harper Valley PTA Theatrical release poster Directed by Richard
Bennett Produced by George Edwards Written by George Edwards Barry Schneider Starring Barbara Eden Nanette Fabray Ronny Cox Louis Nye Susan Swift Pat Paulsen
Music by Nelson Riddle Cinematography Willy Kurant Edited by Michael Economou Distributed by April Fools Productions Release date May 23 , 1978 ( 1978 - 05 - 23
) ( Midwest ) June 2 , 1978 ( 1978 - 06 - 02 ) ( Canada ) August 2 , 1978 ( 1978 - 08 - 02 ) ( Los Angeles ) January 12 , 1979 ( 1979 - 01 - 12 ) ( New York
City ) Running time 93 minutes Country United States Language English Budget $1 million Box office $25 million ( USA ) Harper Valley PTA is a 1978 American
comedy film inspired by the popular 1968 country song `` Harper Valley PTA '' written by Tom T. Hall and performed by country singer Jeannie C. Riley .
</item 10>
=== END ITEM id=e805a9661bf1b476 ===
