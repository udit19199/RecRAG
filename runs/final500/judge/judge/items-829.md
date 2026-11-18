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

=== ITEM id=7737d3a31ce1abe1 ===
Gold sentences and facts:
Sentence 0: This was the city of Chicago 's first professional sports championship since the Chicago Fire won MLS Cup ' 98 ( which came four months after the Chicago Bulls ' sixth NBA championship that year ) . The next major Chicago sports championship came in 2010 , when the NHL 's Chicago Blackhawks ended a 49 - year Stanley Cup title drought . With the Chicago Bears ' win in Super Bowl XX and the Chicago Cubs ' own World Series championship in 2016 , all Chicago sports teams have won at least one major championship since 1985 . Meanwhile , the Astros themselves made it back to the World Series in 2017 , but this time as an AL team , where they defeated the Los Angeles Dodgers in seven games , resulting in Houston 's first professional sports championship since the 2006 -- 07 Houston Dynamo won their back - to - back MLS Championships .
  Fact 0.0: The championship in this context was the first professional sports championship for the city of Chicago since the Chicago Fire won MLS Cup '98.
  Fact 0.1: The Chicago Fire won MLS Cup '98.
  Fact 0.2: The Chicago Fire's MLS Cup '98 win came four months after the Chicago Bulls' sixth NBA championship that year.
  Fact 0.3: The next major Chicago sports championship came in 2010.
  Fact 0.4: In 2010, the NHL's Chicago Blackhawks ended a 49-year Stanley Cup title drought.
  Fact 0.5: The Chicago Bears won Super Bowl XX.
  Fact 0.6: The Chicago Cubs won the World Series in 2016.
  Fact 0.7: All Chicago sports teams have won at least one major championship since 1985.
  Fact 0.8: The Astros made it back to the World Series in 2017.
  Fact 0.9: The Astros were in the 2017 World Series as an American League team.
  Fact 0.10: In 2017, the Astros defeated the Los Angeles Dodgers in seven games in the World Series.
  Fact 0.11: The Astros' 2017 World Series win resulted in Houston's first professional sports championship since the 2006-07 Houston Dynamo won their back-to-back MLS Championships.
  Fact 0.12: The Houston Dynamo won back-to-back MLS Championships in 2006-07.
Sentence 1: Neither team advanced to the post-season in 2006 , but the 2006 World Series again featured teams from the American League Central and National League Central divisions , this time represented by the Detroit Tigers and the St. Louis Cardinals , respectively . The Cardinals won the World Series in five games , in which manager Tony La Russa became the second manager to win the World Series in both American and National leagues , previously managing the Oakland Athletics to the 1989 World Series championship . Both the White Sox and the Astros were in the Wild Card race until the final weeks of the season , with the White Sox finishing with 90 wins , the Astros with 82 wins . The White Sox made their first post-2005 playoff appearance in 2008 , while the Astros would not return to the postseason until 2015 , their third season as an American League team and would not return to the World Series until 2017 , their fifth season as an American League team .
  Fact 1.0: Neither team advanced to the post-season in 2006.
  Fact 1.1: The 2006 World Series again featured teams from the American League Central and National League Central divisions.
  Fact 1.2: The Detroit Tigers and the St. Louis Cardinals represented the American League Central and the National League Central, respectively, in the 2006 World Series.
  Fact 1.3: The St. Louis Cardinals won the World Series in five games.
  Fact 1.4: In the 2006 World Series, manager Tony La Russa became the second manager to win the World Series in both the American and National leagues.
  Fact 1.5: Tony La Russa had previously managed the Oakland Athletics to the 1989 World Series championship.
  Fact 1.6: Both the White Sox and the Astros were in the Wild Card race until the final weeks of the season.
  Fact 1.7: The White Sox finished the season with 90 wins.
  Fact 1.8: The Astros finished the season with 82 wins.
  Fact 1.9: The White Sox made their first post-2005 playoff appearance in 2008.
  Fact 1.10: The Astros would not return to the postseason until 2015, their third season as an American League team.
  Fact 1.11: The Astros would not return to the World Series until 2017, their fifth season as an American League team.

Retrieved items:
<item 1>
2005 World Series
The Astros were attempting to become the fourth consecutive wild card team to win the Series , following the Anaheim Angels ( 2002 ) , Florida Marlins ( 2003 )
and Boston Red Sox ( 2004 ) . Both teams were attempting to overcome decades of disappointment , with a combined 132 years between the two teams without a title
. The Astros were making their first Series appearance in 44 years of play , while the White Sox had waited exactly twice as long for a title , having last won
the Series in 1917 , and had not been in the Series since 1959 , three years before the Astros ' inaugural season . Like the 1982 World Series between the St.
Louis Cardinals and the Milwaukee Brewers , the 2005 World Series is one of only two World Series in the modern era ( 1903 -- present ) with no possibility for
a rematch between the two opponents , because the Astros moved to the American League in 2013 . However , the Brewers did meet the Cardinals in the 2011
National League Championship Series .
</item 1>

<item 2>
2005 World Series
With the Astros on the verge of clinching their first ever National League pennant in Game 5 , Albert Pujols hit a mammoth three - run home run off Brad Lidge
in the top of the ninth inning to take the lead , and subsequently stave off elimination . However , behind NLCS MVP Roy Oswalt , the Astros were able to defeat
the Cards 5 -- 1 in Game 6 and earned a trip to the World Series . This was the Astros ' first World Series appearance in franchise history . Matchups ( edit )
Game 1 ( edit ) Saturday , October 22 , 2005 7 : 05 pm ( CDT ) at U.S.
</item 2>

<item 3>
Houston Astros
Cellular Field in Chicago and the final two in Houston . Game 3 marked the first World Series game held in the state of Texas , and was the longest game in
World Series history , lasting 5 hours and 41 minutes . This World Series was marked by a controversy involving the Minute Maid Park roof . MLB & Commissioner
Bud Selig insisted that the Astros must play with the roof open which mitigated the intensity and enthusiasm of the cheering Astros fans . 2006 -- 2010 : the
decline ( edit ) In the 2006 offseason , the team signed Preston Wilson and moved Berkman to first base , ending the long tenure of Jeff Bagwell . The Astros
renewed the contract with Clemens and traded two minor league prospects to the Tampa Bay Devil Rays for left - handed hitter Aubrey Huff . By August , Preston
Wilson complained about his playing time after the return of Luke Scott from AAA Round Rock . The Astros released Wilson and he was signed by St. Louis . A
dramatic season end included wins in 10 of their last 12 games , but the Astros missed a playoff appearance when they lost the final game of the season to the
Atlanta Braves .
</item 3>

<item 4>
Houston Astros
In 2017 , they became the first franchise in MLB history to have won a pennant in both the NL and the AL , when they defeated the New York Yankees in the ALCS .
They subsequently won the 2017 World Series against the Los Angeles Dodgers , winning four games to three , earning the team , and Texas , its first World
Series title . Contents ( hide ) 1 Franchise history 1.1 Major League Baseball comes to Texas 1.2 1962 -- 1964 : The Colt . 45s 1.3 1965 -- 1970 : The Great
Indoors 1.4 1971 -- 1974 : The boys in orange 1.4.
</item 4>

<item 5>
Graph query row:
opponent: Los Angeles Dodgers
world_series: 2017 World Series Game 1
date: None
</item 5>

<item 6>
2017 World Series
Houston Astros ( edit ) Main article : 2017 Houston Astros season The Astros wore a patch during the 2017 World Series in support of Hurricane Harvey victims in
Houston With a 101 -- 61 regular season record , the team won its first American League West title since moving from the National League Central starting in its
2013 season , and their first division title since 2001 . In the American League Division Series , they defeated the Boston Red Sox in four games and then
defeated the New York Yankees in the American League Championship Series ( ALCS ) in seven games . This was their second World Series appearance and first since
2005 , when they were swept in four games by the Chicago White Sox . They became the first team in history to make it to the World Series as members of both the
National League and the American League . The city of Houston in August 2017 suffered record flooding from Hurricane Harvey . The team began to wear patches
which had the logo of the team with the word `` Strong '' on the bottom of the patch , as well as promoting the hashtag Houston Strong . Manager A.J.
</item 6>

<item 7>
2017 World Series
Contents ( hide ) 1 Background 1.1 Los Angeles Dodgers 1.2 Houston Astros 2 Summary 3 Pre-game ceremonies 4 Game summaries 4.1 Game 1 4.2 Game 2 4.3 Game 3 4.4
Game 4 4.5 Game 5 4.6 Game 6 4.7 Game 7 4.8 Composite line score 5 Broadcasting 5.1 Television 5.1. 1 Ratings 5.2 Radio 6 Sponsorship 7 See also 8 Notes 9
References 10 External links Background ( edit ) This was the first World Series matchup , and second postseason meeting overall , between the Astros and
Dodgers . Los Angeles defeated Houston in the 1981 National League Division Series in five games en route to its World Series championship that year . The teams
also met in the 1980 National League West tie - breaker game . This was the first Fall Classic since 1970 , and the eighth overall , in which both participants
had 100 or more wins during the regular season .
</item 7>

<item 8>
2017 World Series
Team 5 6 7 8 9 10 11 Houston Astros 7 5 34 56 Los Angeles Dodgers 0 9 8 34 47 Home runs : HOU : George Springer ( 5 ) , José Altuve ( 2 ) , Alex Bregman ( 2 ) ,
Carlos Correa ( 2 ) , Yuli Gurriel ( 2 ) , Marwin González ( 1 ) , Brian McCann ( 1 ) LAD : Joc Pederson ( 3 ) , Yasiel Puig ( 2 ) , Cody Bellinger ( 1 ) ,
Charlie Culberson ( 1 ) , Corey Seager ( 1 ) , Chris Taylor ( 1 ) , Justin Turner ( 1 ) Total attendance : 346,702 Average attendance : 49,529 Winning player 's
share : $438,901.57 . Losing player 's share : $259,722.14 .
</item 8>

<item 9>
2017 World Series
In the bottom of the inning , Yasiel Puig hit a home run off of Ken Giles and Enrique Hernández drove in Logan Forsythe to tie the game , with the latter being
the Dodgers ' first run that was not driven in by a home run . In the next inning , George Springer hit a two - run home run for the Astros off of Brandon
McCarthy to retake the lead . In the bottom of the 11th inning , Charlie Culberson homered off of Chris Devenski , who later struck out Puig to end the game .
This was the first ever World Series game in which a team hit home runs in the ninth , tenth and eleventh inning . The teams set a new record for combined home
runs in a single World Series game with eight and this was the first time in MLB history , regular season or postseason , that five home runs were hit in extra
innings . The Astros won their first World Series game in franchise history as they had been swept in their previous appearance in 2005 . Game 3 ( edit ) Lance
McCullers Jr. earned the win in Game 3 .
</item 9>

<item 10>
Houston Astros
They finished 101 -- 61 , with a 21 - game lead in the division , and faced the Red Sox in the first round of the AL playoffs . The Astros defeated the Red Sox
three games to one , and advanced to the American League Championship Series against the New York Yankees . The Astros won the ALCS four games to three , and
advanced to the World Series to play against the Los Angeles Dodgers . The Astros defeated the Dodgers in the deciding seventh game of the World Series ,
winning the first championship in franchise history . The city of Houston celebrated the team 's accomplishment with a parade on the afternoon of November 3 ,
2017 . Houston 's Independent School District gave the students and teachers the day off to watch the parade . On November 16 , 2017 Jose Altuve was named the
American League Most Valuable Player , capping off a historic season in which he accumulated 200 hits for the fourth consecutive season , led the majors with a
. 346 BA , and was the unquestioned clubhouse leader of the World Series Champions .
</item 10>
=== END ITEM id=7737d3a31ce1abe1 ===

=== ITEM id=499694f849f67b80 ===
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
World - renowned artists such as Pablo Picasso , Andy Warhol and Jean - Michel Basquiat helped challenge the status quo , with drawing being very much at the
centre of their practice , and often re-interpreting traditional technique . Basquiat 's drawings were produced in many different mediums , most commonly ink ,
pencil , felt - tip or marker , and oil - stick , and he drew on any surface that came to hand , such as doors , clothing , refrigerators , walls and baseball
helmets . Modern - day artists such as Tracey Emin observe the legacy and importance of drawing , engraving and print making techniques in their vast array ,
passed down through millennia .
</item 1>

<item 2>
Drawing
At this point , drawing was commonly used as a tool for thought and investigation , acting as a study medium whilst artists were preparing for their final
pieces of work . The Renaissance brought about a great sophistication in drawing techniques , enabling artists to represent things more realistically than
before , and revealing an interest in geometry and philosophy . The invention of the first widely available form of photography led to a shift in the hierarchy
of the arts . Photography offered an alternative to drawing as a method for accurately representing visual phenomena , and traditional drawing practice was
given less emphasis as an essential skill for artists , particularly so in Western society . Example of a typical drawing by American artist and notable
draftsman , Jean - Michel Basquiat , Untitled ( Axe / Rene ) 1982 . Basquiat produced over 1500 drawings and pushed the boundaries of drawing in the late 20th -
century Notable artists and draftsmen ( edit ) Drawing became significant as an art form around the late 15th century , with artists and master engravers such
as Albrecht Dürer and Martin Schongauer ( c. 1448 - 1491 ) , the very first Northern engraver known by name .
</item 2>

<item 3>
Drawing
Jump up ^ Ruskin , John ( 1857 ) . The Elements of Drawing . Mineola , NY : Dover Publishcations Inc . Jump up ^ McManus , I.C. ; Chamberlain , R.S. ; Loo , P.
- K. ; Rankin , Q. ; Riley , H. ; Brunswick , N. ( 2010 ) . `` Art students who can not draw : exploring the relations between drawing ability , visual memory ,
accuracy of copying , and dyslexia '' ( PDF ) . Psychology of Aesthetics , Creativity and the Arts . 4 : 18 -- 30 . doi : 10.1037 / a0017335 . Jump up ^ Fayena
- Tawil , F. ; Kozbelt , A. ; Sitaras , S. ( 2011 ) . `` Think global , act local : A protocol analysis comparison of artists ' and nonartists ' cognitions ,
metacognitions , and evaluations while drawing '' . Psychology of Aesthetics , Creativity and the Arts . 5 : 135 -- 45 .
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
Pen and ink drawings often use hatching -- groups of parallel lines . Cross-hatching uses hatching in two or more different directions to create a darker tone .
Broken hatching , or lines with intermittent breaks , form lighter tones -- and controlling the density of the breaks achieves a gradation of tone . Stippling
uses dots to produce tone , texture and shade . Different textures can be achieved depending on the method used to build tone . Drawings in dry media often use
similar techniques , though pencils and drawing sticks can achieve continuous variations in tone . Typically a drawing is filled in based on which hand the
artist favors . A right - handed artist draws from left to right to avoid smearing the image . Erasers can remove unwanted lines , lighten tones , and clean up
stray marks . In a sketch or outline drawing , lines drawn often follow the contour of the subject , creating depth by looking like shadows cast from a light in
the artist 's position . Sometimes the artist leaves a section of the image untouched while filling in the remainder .
</item 5>

<item 6>
Drawing
Escher Arshile Gorky George Grosz Paul Klee Oscar Kokoschka Käthe Kollwitz Alfred Kubin André Masson Alphonse Mucha Jules Pascin Pablo Picasso Egon Schiele Jean
- Michel Basquiat Andy Warhol Frank Auerbach 21st : Tracey Emin Banksy Materials ( edit ) The medium is the means by which ink , pigment or color are delivered
onto the drawing surface . Most drawing media are either dry ( e.g. graphite , charcoal , pastels , Conté , silverpoint ) , or use a fluid solvent or carrier (
marker , pen and ink ) . Watercolor pencils can be used dry like ordinary pencils , then moistened with a wet brush to get various painterly effects . Very
rarely , artists have drawn with ( usually decoded ) invisible ink . Metalpoint drawing usually employs either of two metals : silver or lead . More rarely used
are gold , platinum , copper , brass , bronze , and tinpoint . Paper comes in a variety of different sizes and qualities , ranging from newspaper grade up to
high quality and relatively expensive paper sold as individual sheets .
</item 6>

<item 7>
Drawing
The wide availability of drawing instruments makes drawing one of the most common artistic activities . In addition to its more artistic forms , drawing is
frequently used in commercial illustration , animation , architecture , engineering and technical drawing . A quick , freehand drawing , usually not intended as
a finished work , is sometimes called a sketch . An artist who practices or works in technical drawing may be called a drafter , draftsman or a draughtsman .
Contents ( hide ) 1 Overview 2 History 2.1 Drawing in communication 2.2 Drawing in manuscripts 2.3 Drawing in science 2.4 Drawing as artistic expression 3
Notable artists and draftsmen 4 Materials 5 Technique 6 Tone 7 Form and proportion 8 Perspective 9 Artistry 10 Process 11 Seven Ps of drawing 12 See also 13
References 14 External links Overview ( edit ) Galileo Galilei . Phases of the Moon . 1616 . Drawing is one of the major forms of expression within the visual
arts . It is generally concerned with the marking of lines and areas of tone onto paper / other material , where the accurate representation of the visual world
is expressed upon a plane surface .
</item 7>

<item 8>
Drawing
There are also many drawing methods , such as line drawing , stippling , shading , the surrealist method of entopic graphomania ( in which dots are made at the
sites of impurities in a blank sheet of paper , and lines are then made between the dots ) , and tracing ( drawing on a translucent paper , such as tracing
paper , around the outline of preexisting shapes that show through the paper ) . A quick , unrefined drawing may be called a sketch . In fields outside art ,
technical drawings or plans of buildings , machinery , circuitry and other things are often called `` drawings '' even when they have been transferred to
another medium by printing . History ( edit ) Drawing in communication ( edit ) Drawing is one of the oldest forms of human expression , with evidence for its
existence preceding that of written communication . It is believed that drawing was used as a specialised form of communication before the invention of the
written language , demonstrated by the production of cave and rock paintings around 30,000 years ago ( Art of the Upper Paleolithic ) . These drawings , known
as pictograms , depicted objects and abstract concepts .
</item 8>

<item 9>
Drawing
Drawing - wikipedia Drawing Jump to : navigation , search For other uses , see Drawing ( disambiguation ) . Pen and wash lion by Rembrandt in the Louvre Drawing
is a form of visual art in which a person uses various drawing instruments to mark paper or another two - dimensional medium . Instruments include graphite
pencils , pen and ink , inked brushes , wax colored pencils , crayons , charcoal , chalk , pastels , various kinds of erasers , markers , styluses , various
metals ( such as silverpoint ) and electronic drawing . A drawing instrument releases a small amount of material onto a surface , leaving a visible mark . The
most common support for drawing is paper , although other materials , such as cardboard , plastic , leather , canvas , and board , may be used . Temporary
drawings may be made on a blackboard or whiteboard or indeed almost anything . The medium has been a popular and fundamental means of public expression
throughout human history . It is one of the simplest and most efficient means of communicating visual ideas . The wide availability of drawing instruments makes
drawing one of the most common artistic activities .
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
=== END ITEM id=499694f849f67b80 ===

=== ITEM id=54c8f2b500f2a925 ===
Gold sentences and facts:
Sentence 0: 12 -- Bart Cummings ( 1965 , 1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 )
  Fact 0.0: Bart Cummings has 12 Melbourne Cup wins.
  Fact 0.1: Bart Cummings won the Melbourne Cup in 1965.
  Fact 0.2: Bart Cummings won the Melbourne Cup in 1966.
  Fact 0.3: Bart Cummings won the Melbourne Cup in 1967.
  Fact 0.4: Bart Cummings won the Melbourne Cup in 1974.
  Fact 0.5: Bart Cummings won the Melbourne Cup in 1975.
  Fact 0.6: Bart Cummings won the Melbourne Cup in 1977.
  Fact 0.7: Bart Cummings won the Melbourne Cup in 1979.
  Fact 0.8: Bart Cummings won the Melbourne Cup in 1990.
  Fact 0.9: Bart Cummings won the Melbourne Cup in 1991.
  Fact 0.10: Bart Cummings won the Melbourne Cup in 1996.
  Fact 0.11: Bart Cummings won the Melbourne Cup in 1999.
  Fact 0.12: Bart Cummings won the Melbourne Cup in 2008.
Sentence 1: 12 -- Bart Cummings ( 1965 , 1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 ) 5 -- Etienne de Mestre ( 1861 , 1862 , 1867 , 1877 , 1878 ) 5 -- Lee Freedman ( 1989 , 1992 , 1995 , 2004 , 2005 )
  Fact 1.0: Bart Cummings has 12 Melbourne Cup wins.
  Fact 1.1: Bart Cummings won the Melbourne Cup in 1965.
  Fact 1.2: Bart Cummings won the Melbourne Cup in 1966.
  Fact 1.3: Bart Cummings won the Melbourne Cup in 1967.
  Fact 1.4: Bart Cummings won the Melbourne Cup in 1974.
  Fact 1.5: Bart Cummings won the Melbourne Cup in 1975.
  Fact 1.6: Bart Cummings won the Melbourne Cup in 1977.
  Fact 1.7: Bart Cummings won the Melbourne Cup in 1979.
  Fact 1.8: Bart Cummings won the Melbourne Cup in 1990.
  Fact 1.9: Bart Cummings won the Melbourne Cup in 1991.
  Fact 1.10: Bart Cummings won the Melbourne Cup in 1996.
  Fact 1.11: Bart Cummings won the Melbourne Cup in 1999.
  Fact 1.12: Bart Cummings won the Melbourne Cup in 2008.
  Fact 1.13: Etienne de Mestre has 5 Melbourne Cup wins.
  Fact 1.14: Etienne de Mestre won the Melbourne Cup in 1861.
  Fact 1.15: Etienne de Mestre won the Melbourne Cup in 1862.
  Fact 1.16: Etienne de Mestre won the Melbourne Cup in 1867.
  Fact 1.17: Etienne de Mestre won the Melbourne Cup in 1877.
  Fact 1.18: Etienne de Mestre won the Melbourne Cup in 1878.
  Fact 1.19: Lee Freedman has 5 Melbourne Cup wins.
  Fact 1.20: Lee Freedman won the Melbourne Cup in 1989.
  Fact 1.21: Lee Freedman won the Melbourne Cup in 1992.
  Fact 1.22: Lee Freedman won the Melbourne Cup in 1995.
  Fact 1.23: Lee Freedman won the Melbourne Cup in 2004.
  Fact 1.24: Lee Freedman won the Melbourne Cup in 2005.
Sentence 2: 1877 is also the year that the trainer Etienne de Mestre won his fourth Melbourne Cup with Chester owned by Hon . James White . In 1878 , as in previous years De Mestre fielded more than one horse . He entered the favourite Firebell ( owned by W.S. Cox ) who finished last , Chester ( owned by Hon . James White ) the previous year 's winner who fell , and Calamia ( owned by de Mestre ) who , though less fancied , won easily by two lengths . First prize was £ 1,790 , the crowd was 80,000 and there were 30 starters . De Mestre 's 1878 win with Calamia brought to 5 the number of Melbourne Cups he had won . This record was not to be matched for nearly 100 years when the trainer Bart Cummings won his fifth Melbourne Cup in 1975 . Bart Cummings , regarded as the best Australian horse trainer of all time , went on to win 12 Melbourne Cups to 2008 .
  Fact 2.0: In 1877, the trainer Etienne de Mestre won his fourth Melbourne Cup.
  Fact 2.1: In 1877, Etienne de Mestre won with Chester.
  Fact 2.2: Chester was owned by Hon. James White.
  Fact 2.3: In 1878, as in previous years, Etienne de Mestre fielded more than one horse.
  Fact 2.4: Etienne de Mestre entered the favourite Firebell in 1878.
  Fact 2.5: Firebell was owned by W.S. Cox.
  Fact 2.6: Firebell finished last in 1878.
  Fact 2.7: Etienne de Mestre entered Chester in 1878.
  Fact 2.8: Chester was owned by Hon. James White.
  Fact 2.9: Chester was the previous year's winner.
  Fact 2.10: Chester fell in 1878.
  Fact 2.11: Etienne de Mestre entered Calamia in 1878.
  Fact 2.12: Calamia was owned by Etienne de Mestre.
  Fact 2.13: Calamia was less fancied.
  Fact 2.14: Calamia won easily by two lengths in 1878.
  Fact 2.15: First prize in 1878 was £1,790.
  Fact 2.16: The crowd in 1878 was 80,000.
  Fact 2.17: There were 30 starters in 1878.
  Fact 2.18: De Mestre's 1878 win with Calamia brought to 5 the number of Melbourne Cups he had won.
  Fact 2.19: De Mestre's record of 5 Melbourne Cups was not matched for nearly 100 years.
  Fact 2.20: Bart Cummings won his fifth Melbourne Cup in 1975.
  Fact 2.21: Bart Cummings is regarded as the best Australian horse trainer of all time.
  Fact 2.22: Bart Cummings went on to win 12 Melbourne Cups, up to 2008.

Retrieved items:
<item 1>
Melbourne Cup
1 Melbourne Cup Tour 3 History 3.1 Early years 3.2 Recent years 3.3 Timeline of notable events 3.4 Public holiday 4 Results and records 4.1 Most wins by a horse
4.2 Most wins by a jockey 4.3 Most wins by a trainer 4.4 Most wins by an owner 4.5 Other records 5 Attendance 6 Off the track 7 See also 8 Notes 9 Further
reading 10 External links Qualifying and race conditions ( edit ) 1932 & 1934 winner Peter Pan ridden by jockey Darby Munro , won twice The race is a quality
handicap for horses 3 years old and over , run over a distance of 3,200 metres , on the first Tuesday in November at Flemington Racecourse . The minimum
handicap weight is 50 kg . There is no maximum weight , but the top allocated weight must not be less than 57 kg . The weight allocated to each horse is
declared by the VRC Handicapper in early September .
</item 1>

<item 2>
Melbourne Cup
Perhaps they should have kept their money , however . The outsider Briseis comfortably won by 1 length in the biggest field of all time , and in an extremely
good time , so it is unlikely that the horses who perished could have beaten her . 1877 is also the year that the trainer Etienne de Mestre won his fourth
Melbourne Cup with Chester owned by Hon . James White . In 1878 , as in previous years De Mestre fielded more than one horse . He entered the favourite Firebell
( owned by W.S. Cox ) who finished last , Chester ( owned by Hon . James White ) the previous year 's winner who fell , and Calamia ( owned by de Mestre ) who ,
though less fancied , won easily by two lengths . First prize was £ 1,790 , the crowd was 80,000 and there were 30 starters . De Mestre 's 1878 win with Calamia
brought to 5 the number of Melbourne Cups he had won . This record was not to be matched for nearly 100 years when the trainer Bart Cummings won his fifth
Melbourne Cup in 1975 .
</item 2>

<item 3>
Melbourne Cup
Both horses had been stabled in Japan . Corowa , NSW trained `` Leica Falcon '' also was not be permitted to race in Victoria , despite Corowa being close to
the Victorian border . Leica Falcon was ordained as the new staying star of Australian racing in 2005 when he ran fourth in both the Caulfield Cup and in Makybe
Diva 's famous third Melbourne Cup victory . But serious leg injuries saw the horse not race for another 20 months . Efficient , the previous year 's VRC Derby
winner , won the race . In 2013 , Damien Oliver returned from an eight - month ban , after betting against his own mount at a previous race meet , to win his
3rd Melbourne cup . 180 degree panorama of the racecourse Timeline of Notable events ( edit ) 1861 - The very first Melbourne Cup was held with 17 runners . The
eventful race ( a horse bolted prior to the start and 3 horses fell during the race of which 2 had to be put down ) was won by Archer . 1862 - Archer became the
first horse to win the race twice .
</item 3>

<item 4>
Melbourne Cup
He carried 53 lb ( 24 kg ) more than second placed horse Highborn . 1894 -- Strand starts were introduced to Flemington . 1896 -- The Melbourne Cup was first
filmed . This race was won by Newhaven . 1915 -- First woman owner to win was Mrs E.A. Widdis with Patrobas . 1916 -- The race was postponed until the following
Saturday . 1925 -- The first radio broadcast of the Melbourne Cup was made by the Australian Broadcasting Company . 1930 - The legendary Phar Lap won his
Melbourne Cup ( after a 3rd placing the previous year ) , as the shortest priced favourite in history and the only favourite to win at ' odds on ' ( 8 / 11 ) .
1931 -- The first year the totalisator operated at the Melbourne Cup . The Totalisator Agency Board was introduced in 1961 . It was also the last of 3 Melbourne
Cups that Phar Lap contested , where the handicappers allotted him 10 st 10 lb ( 68 kg ) , the heaviest weight ever carried in the race . He only managed to
finish 8th behind White Nose .
</item 4>

<item 5>
Melbourne Cup
Subsequent foreign bred horses to win Cup were Backwood 1924 ; Phar Lap 1930 ; Wotan 1936 Belldale Ball 1980 ; At Talaq 1986 ; Kingston Rule 1990 ; Vintage Crop
1993 ; Jeune 1994 ; Media Puzzle 2002 ; Makybe Diva 2003 , 2004 , 2005 ; Americain 2010 and Dunaden 2011 . The 1938 Melbourne Cup was won by trainer Mrs. Allan
McDonald , who conditioned Catalogue . Mrs McDonald was a successful trainer in New Zealand , however , at the time women were not allowed to compete as
trainers in Australia so her husband 's name was officially recorded as the winning trainer . The 2001 edition was won by New Zealand mare Ethereal , trained by
Sheila Laxon , the first woman to formally train a Melbourne Cup winner . She also won the Caulfield Cup , a 2,400 metre race also held in Melbourne , and
therefore has won the `` Cups Double '' .
</item 5>

<item 6>
Melbourne Cup
The first jockey of Indigenous heritage to ride a Melbourne Cup winner was Frank Reys in 1973 on Gala Supreme , who had a Filipino father and a half -
Aboriginal mother . Recent years ( edit ) Wikinews has related news : Delta Blues wins 2006 Melbourne Cup Makybe Diva wins the Melbourne Cup Efficient wins
Melbourne Cup Viewed wins the 2008 Melbourne Cup The race has undergone several alterations in recent years , the most visible being the entry of many foreign -
trained horses . Most have failed to cope with the conditions ; the three successful `` foreign raids '' include two by Irish trainer Dermot K. Weld successful
in 1993 and 2002 , and one in 2006 by Katsumi Yoshida of Japan 's renowned Yoshida racing and breeding family . The attraction for foreigners to compete was ,
primarily , the low - profile change to the new `` quality handicap '' weighting system . The 1910 Melbourne Cup was won by Comedy King , the first foreign bred
horse to do so .
</item 6>

<item 7>
Melbourne Cup
Bart Cummings , regarded as the best Australian horse trainer of all time , went on to win 12 Melbourne Cups to 2008 . Martini - Henry , the 1883 Melbourne Cup
winner In 1883 , the hardy New Zealand bred , Martini - Henry won the VRC Derby , the Melbourne Cup and on the following Monday retained his undefeated record
by winning Mares ' Produce Stakes . Phar Lap winning the Melbourne Cup Race from Second Wind and Shadow King on 5 November 1930 . Phar Lap , the most famous
horse in the world of his day , won the 1930 Melbourne Cup at 11 / 8 odds on , the shortest priced favourite in the history of the race . He had to be hidden
away at Geelong before the race after an attempt was made to shoot him and only emerged an hour before the race time of the Cup . Phar Lap also competed in 1929
and 1931 , but came 3rd and 8th respectively , despite heavy favouritism in both years . There are a few legends of the first Aboriginal jockey to ride a
Melbourne Cup .
</item 7>

<item 8>
Melbourne Cup
Maree Lyndon became the first female to ride in the Melbourne Cup , when she partnered Argonaut Style in 1987 , in which she ran second last in the 21 horse
field . In 2004 , Makybe Diva became the first mare to win two cups , and also the first horse to win with different trainers , after David Hall moved to Hong
Kong and transferred her to the Lee Freedman stables . The 2005 Melbourne Cup was held before a crowd of 106,479 . Makybe Diva made history by becoming the only
horse to win the race three times . Trainer Lee Freedman said after the race , `` Go and find the youngest child on the course , because that 's the only person
here who will have a chance of seeing this happen again in their lifetime . '' Due to the 2007 Australian Equine influenza outbreak , believed to have been
started by a horse brought into Australia from Japan , neither Delta Blues nor Pop Rock participated in the 2007 Melbourne Cup . Both horses had been stabled in
Japan .
</item 8>

<item 9>
Melbourne Cup
The race was won by the George Hanlon trained 40 - 1 outsider Piping Lane ridden by John Letts and carrying only 48 kg . 1985 -- The first sponsored Melbourne
Cup , and the first million dollar Cup , with $650,000 for the winner . 1987 -- First female jockey to ride in the cup was Maree Lyndon on Argonaut Style .
Dunaden : 2011 Melbourne Cup winner , painted by Charles Church 1993 -- The Dermot Weld trained Irish gelding Vintage Crop became the first Northern hemisphere
trained horse to win the Melbourne Cup and the first Irish horse . 2001 -- Sheila Laxon , was the first woman trainer to officially win the Melbourne Cup .
However , Mrs. A. McDonald ( 1938 ) with Catalogue was really the first woman trainer to win . Women then could not be registered as trainers in Australia , and
it was her husband who was the registered trainer . Mrs. McDonald 's win was as a female trainer of a female owned horse . 2003 -- First Australian female
jockey to ride in the cup was Clare Lindop on Debben .
</item 9>

<item 10>
Melbourne Cup
2003 -- The first Melbourne Cup Tour was conducted around Australia , and the biggest crowd , of 122,736 , is recorded at Flemington . 2005 -- Makybe Diva
became the only horse so far to win the Melbourne Cup three times . 2008 -- The `` Cup King '' , Bart Cummings , took his 12th win in the Melbourne Cup with
Viewed 2010 - 150th anniversary . Americain becomes the first French - trained horse to win the race , and Gerald Mosse the first French jockey . 2011 - French
based horse Dunaden wins from Red Cadeaux in the closest finish ever . 2012 - The first seven finishers in the race were bred in Ireland . 2013 - Trainer Gai
Waterhouse became the first Australian female trainer to train the winner of the race when Fiorente won . Her father , legendary trainer Tommy J. Smith was also
a Melbourne Cup Winning trainer ( Toparoa in 1955 and Just A Dash in 1981 ) . 2014 - Two horses die due to racing in the 2014 Melbourne Cup .
</item 10>
=== END ITEM id=54c8f2b500f2a925 ===

=== ITEM id=5fe96c61460ef239 ===
Gold sentences and facts:
Sentence 0: Reading Full name Reading Football Club Nickname ( s ) The Royals Founded 1871 ; 147 years ago ( 1871 ) Ground Madejski Stadium Ground Capacity 24,161 Owner Dai Yongge and Dai Xiuli ( majority ) Chairman Sir John Madejski Manager Jaap Stam League Championship 2016 -- 17 Championship , 3rd Website Club website Home colours Away colours Current season
  Fact 0.0: The full name of the club is Reading Football Club.
  Fact 0.1: The club's nickname is The Royals.
  Fact 0.2: Reading was founded in 1871.
  Fact 0.3: Reading was founded 147 years ago.
  Fact 0.4: Reading's ground is Madejski Stadium, with a ground capacity of 24,161.
  Fact 0.5: Reading's owners are Dai Yongge and Dai Xiuli (majority).
  Fact 0.6: Reading's chairman is Sir John Madejski.
  Fact 0.7: Reading's manager is Jaap Stam.
  Fact 0.8: Reading competed in the Championship in the 2016-17 season, finishing 3rd.
  Fact 0.9: Reading's website is listed as the club website.
Sentence 1: Reading Football Club ownership structure : 75 % Owned by Renhe Sports Management Ltd , 100 % owned by Xiu Li Dai and Yongge Dai . 25 % Owned by Narin Niruttinanon
  Fact 1.0: Reading Football Club's ownership structure is 75% owned by Renhe Sports Management Ltd.
  Fact 1.1: Renhe Sports Management Ltd is 100% owned by Xiu Li Dai and Yongge Dai.
  Fact 1.2: 25% of Reading Football Club is owned by Narin Niruttinanon.

Retrieved items:
<item 1>
Reading F.C.
On 24 May 2007 , it was announced that planning permission had been granted to extend the stadium to a capacity of 36,900 . Reading have made plans for a new
training ground at Bearwood Golf Club to replace Hogwood park their current training facility . On 5 July 2016 , at the end of Eamonn Dolan 's funeral , Reading
announced that the North Stand would be renamed the `` Eamonn Dolan Stand '' . Support ( edit ) See also : Supporters ' Trust at Reading Supporters at a Reading
match at Elm Park in 1913 In 2001 , Reading became the first football club to register their fans as an official member of their squad , giving the `` player ''
registered with squad number 13 as ' Reading Fans ' . Rivalries ( edit ) Before going out of business in 1992 , Aldershot were Reading 's biggest rivals . There
was a strong rift between the two sets of fans , with fighting between fans occurring on several occasions . Strong feelings remain between fans of Reading and
fans of Aldershot Town , the refounded club in Aldershot .
</item 1>

<item 2>
Reading F.C.
. Official website Supporters ' Trust At Reading Reading F.C. on BBC Sport : Club news -- Recent results -- Upcoming fixtures Reading Football Club Current
season Under - 21s & Academy Records & Statistics Honours History History Seasons Thames Valley Royals proposal Grounds Elm Park ( 1896 -- 1998 ) Madejski
Stadium ( 1998 -- present ) People Managers Players with 1 -- 24 appearances 25 -- 99 appearances 100 + appearances Internationals Player of the Season Notable
matches Level 2 play - off finals 1995 2011 2017 Level 3 play - off finals Full Members ' Cup finals 1988 Other matches Portsmouth 7 -- 4 Reading Related
articles Hob Nob Anyone ? Supporters ' Trust At Reading Reading Women Marussia F1 Book : Reading F.C. Category : Reading F.C. Portal : Association football
Commons : Reading F.C. Reading F.C. seasons 1940 -- 41 1941 -- 42 1942 -- 43 1943 -- 44 1944 -- 45 1945 -- 46 1946 -- 47 1947 -- 48 1948 -- 49
</item 2>

<item 3>
Graph query row:
club: Reading Football Club
person: Dai Yongge
relationship: RELATED_TO
role: majority owner
relationship_detail: None
relation: None
fact: None
context: None
</item 3>

<item 4>
Reading F.C.
Reading were promoted to English football 's top division for the first time in their history . The 2006 -- 07 season saw Reading make their first appearance in
the top flight of English football . Reading defied pre-season predictions of relegation to finish the season in eighth place with 55 points . Reading turned
down the chance to play in the UEFA Intertoto Cup . In the run up to their second season in the Premier League , Reading took part in the 2007 Peace Cup in
South Korea . This second season was less successful , however , and Reading were relegated back to the Championship . Reading started the 2008 -- 09 season
with a 15 match unbeaten home run . They finished fourth and qualify for the play - offs , where they lost to Burnley in the semi-final . Manager Steve Coppell
resigned just hours after the game , replaced by Brendan Rodgers . Rodgers left the club by mutual consent on 16 December 2009 and Brian McDermott made
caretaker manager the same day .
</item 4>

<item 5>
Reading F.C.
internationals Former players ( edit ) Main article : List of Reading F.C. players Club officials ( edit ) Board of Directors & Senior Club Staff Chief
Executive : Ron Gourlay Commercial Director : Adam Benson Club Secretary : Sue Hewett Financial Controller : Bryan Stabler Directors : Sir John Madejski ( Co
Chairman ) , Sasima Srivikorn ( Co Chairwoman ) , Nigel Howe , Narin Niruttinanon , Sumrith Thanakarnjanasuth , Taweesuk Srisumrid and Theekharoj Piamphongsarn
Management team Director of Football : Brian Tevreden First Team Manager : Jaap Stam Assistant Manager : Andries Ulderink & Said Bakkati First Team Coach :
Michael Gilkes Goalkeeping Coach : Dave Beasant First Team Sports Scientist : Tristan Baker First Team Physio : Jon Urwin Academy Manager : Lee Herron Assistant
Academy Manager : David Dodds Academy Loan Manager : Mikele Leigertwood Kit Manager : Stewart Bannister First Team Head Scout : Steve Shorey Masseur : Dan
Buchanan Managers ( edit ) Main article : List of Reading F.C.
</item 5>

<item 6>
Reading F.C.
Reading then finished eighth in the 2006 -- 07 Premier League , their first ever season as a top flight club . Contents ( hide ) 1 History 2 Crest and colours 3
Stadium 4 Support 4.1 Rivalries 5 Sponsorship 5.1 Additional kit sponsors 6 Ownership and finances 7 Players 7.1 First - team squad 7.2 Under - 23 squad 7.3
Under - 18 squad 7.4 Out on loan 8 Records and statistics 8.1 Notable players 8.2 Player of the season 8.3 International players 8.4 Former players 9 Club
officials 10 Managers 11 Women 's team 12 Honours 12.1 Managerial 13 Affiliated clubs 14 References 15 Bibliography 16 External links History ( edit ) Main
article : History of Reading F.C. See also : Thames Valley Royals proposal Reading were formed on 25 December 1871 , following a public meeting at the Bridge
Street Rooms organised by the future club secretary Joseph Edward Sydenham .
</item 6>

<item 7>
Reading F.C.
Reading FC - Wikipedia Reading FC Jump to : navigation , search Not to be confused with Reading F.C. Women . Reading Full name Reading Football Club Nickname (
s ) The Royals Founded 1871 ; 147 years ago ( 1871 ) Ground Madejski Stadium Ground Capacity 24,161 Owner Dai Yongge and Dai Xiuli ( majority ) Chairman Sir
John Madejski Manager Jaap Stam League Championship 2016 -- 17 Championship , 3rd Website Club website Home colours Away colours Current season Reading Football
Club ( / ˈrɛdɪŋ / ( listen ) RED - ing ) is a professional association football club based in Reading , Berkshire , England . The team play in the Championship
, the second tier of English football . Reading are nicknamed The Royals , due to Reading 's location in the Royal County of Berkshire , though they were
previously known as The Biscuitmen , due to the town 's association with Huntley and Palmers .
</item 7>

<item 8>
Reading F.C.
25 % Owned by Narin Niruttinanon Reading F.C. Directors : Xiu Li Dai , Yongge Dai , Ronald Gourlay , Narin Niruttinanon , Nigel Howe Players ( edit ) As of 8
January 2018 First - team squad ( edit ) Note : Flags indicate national team as defined under FIFA eligibility rules . Players may hold more than one non-FIFA
nationality . No . Position Player GK Vito Mannone DF Chris Gunter MF Joey van den Berg 5 DF Paul McShane ( captain ) 6 DF Liam Moore 7 MF Roy Beerens 8 MF
George Evans 9 FW Joseph Mendes 10 MF John Swift 11 DF Jordan Obita 12 MF Garath McCleary 14 FW Sone Aluko 15 MF Callum Harriott 16 MF David Edwards 17 FW Modou
Barrow 18 FW Yann Kermorgant 19 MF Leandro Bacuna No .
</item 8>

<item 9>
Graph query row:
club: Reading Football Club
person: Dai Xiuli
relationship: RELATED_TO
role: majority owner
relationship_detail: None
relation: None
fact: None
context: None
</item 9>

<item 10>
Reading F.C.
Retrieved on 17 July 2013 . Jump up ^ HOW SOUTH GEORGIA UNITED BECAME A READING FC AFFILIATE CLUB . South Georgia United . Retrieved on 5 June 2014 .
Bibliography ( edit ) Digby , Bob ( 2001 ) . It 's a World Thing . Oxford , Oxfordshire : Oxford University Press . ISBN 978 - 0 - 19 - 913428 - 1 . Retrieved 2
June 2011 . Yeo , Stephen ( 1976 ) . Religion and Voluntary Organisations in Crisis . Abingdon , Oxfordshire : Taylor & Francis . ISBN 978 - 0 - 85664 - 017 - 9
. Retrieved 1 June 2011 . External links ( edit ) Wikimedia Commons has media related to Reading F.C. . Official website Supporters ' Trust At Reading Reading
F.C.
</item 10>
=== END ITEM id=5fe96c61460ef239 ===

=== ITEM id=dd335896c1730d62 ===
Gold sentences and facts:
Sentence 0: No . Cloverway inc . title Original Japanese and Viz Media titles Episode director ( s ) Writer ( s ) Art director ( s ) Animation director ( s ) Original airdate English airdate ( United States ) 90 `` Star Struck , Bad Luck '' `` Premonition of the Apocalypse : The Mysterious New Guardians Appear '' `` Chikyū hōkai no yokan ? Nazo no shin Senshi shutsugen '' ( 地球 崩壊 の 予感 ? 謎 の 新 戦士 出現 ) Junichi Sato Sukehiro Tomita Kenichi Tajiri Masahiro Andō March 19 , 1994 June 12 , 2000 After experiencing apocalyptic visions , Rei is attacked by a powerful monster called a Daimon that Sailor Moon is unable to fight after it disables her brooch , reversing her transformation . The Daimon is defeated by a pair of mysterious new warriors , leaving the girls to despondently realize that a new conflict has begun . 91 `` Crystal Clear Again '' `` The Rod of Love Is Born : Usagi 's New Transformation '' `` Ai no roddo tanjō ! Usagi no shin henshin '' ( 愛 の ロッド 誕生 ! うさぎ の 新 変身 ) Yūji Endō Shigeru Yanagawa Yoshiyuki Shikano Katsumi Tamegai March 26 , 1994 June 13 , 2000 While Mamoru strives to cheer Usagi up after her failure in the previous battle , a young girl is the latest target of the Death Busters . Due to Usagi 's inability to transform , the under - powered Sailor Team struggles to fight off the Daimon and are once again forced to rely on the mysterious two warriors . Once they determine the girl 's pure heart is not a talisman , they leave the Sailor Team to fend for themselves . Meanwhile , Tuxedo Mask and Usagi hold hands after the latter 's refusal to escape , and through the power of their love , Usagi 's disabled Crystal Star brooch transforms into the Cosmic Heart Compact , and she gains the Spiral Heart Moon Rod and , along with it , the ability to transform and defeat the new enemy . 92 `` Driving Dangerously '' `` A Handsome Boy ? Haruka Tenoh 's Secret '' `` Suteki na bishōnen ? Ten'ō Haruka no himitsu '' ( 素敵 な 美少年 ? 天王 はるか の 秘密 ) Kunihiko Ikuhara Katsuyuki Sumisawa Kazuhisa Asai Hisashi Kagawa April 16 , 1994 June 14 , 2000 Avoiding their study group , Usagi and Minako meet Haruka Tenou , an attractive and supposedly male student of a prestigious local school , with whom they both become infatuated . After Minako is defeated in a racing game , Michiru arrives to pick up Haruka , and the two depart with Usagi and Minako soon following behind them . Once Michiru and Haruka separate at a mechanic 's shop , Usagi and Minako take this as their chance , and Haruka tells them about `` him '' - self until a Daimon arrives to attack Kameda , the shop mechanic . The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down , when Uranus and Neptune arrive to investigate Kameda 's pure heart crystal . When it turns out not to be a talisman they retreat and Sailor Moon defeats the Daimon . Later , the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they 'd told them about ; Michiru arrives with a blonde girl , whom Usagi and Minako initially fail to realize is Haruka ; once they do , Minako falls to the ground in anguish , and Haruka notes that she 'd never said she was male . 93 `` Bad Harmony '' `` Usagi 's Idol : The Graceful Genius Michiru '' `` Usagi no akogare ! Yūbi no tensai Michiru '' ( うさぎ の 憧れ ! 優美 な 天才 みちる ) Hiroki Shibata Megumi Sugihara Kazuyuki Hashimoto Taichi Nakamura April 23 , 1994 June 15 , 2000 When Usagi is concerned that she is not smart enough to be with Mamoru after being scolded by the other girls , she seeks advice from Michiru Kaioh , who suggests that Usagi and Mamoru go to a violin recital . 94 `` Swept Off Her Feet '' `` Protect the Pure Heart : The Three - Way Battle '' `` Pyua na kokoro wo mamore ! Teki mikata mittsu tomoe ransen '' ( 純 な 心 を 守れ ! 敵 味方 三つ巴 乱戦 ) Noriyo Sasaki Sukehiro Tomita Kenichi Tajiri Ikuko Itō April 30 , 1994 June 16 , 2000 Usagi 's friend , Unazuki , is attacked by a Daimon and has her pure heart crystal stolen . But when Sailor Moon tries to take the crystal back to Unazuki , Uranus and Neptune attempt to steal it in order to claim the Talisman . 95 `` Blinded By Love 's Light '' `` Let Moon Help With Your Love Problems '' `` Koi no otasuke wa Mūn ni omakase '' ( 恋 の おたすけ は ムーン に おまかせ ) Harume Kosaka Shigeru Yanagawa Yoshiyuki Shikano Mari Tominaga May 7 , 1994 June 19 , 2000 Naru and Umino enter an affection contest in order to prove their strong mutual romantic love for each other , but find themselves targeted by a Daimon . 96 `` Lita Borrows Trouble '' `` Coldhearted Uranus : Makoto in Danger '' `` Reikoku na Uranusu ? Makoto no pinchi '' ( 冷酷 な ウラヌス ? まこと の ピンチ ) Kōnosuke Uda Yoji Enokido Kazuhisa Asai Masahiro Andō May 14 , 1994 June 20 , 2000 Makoto is targeted by a Daimon , but manages to temporarily evade it . Keen to analyze her pure heart crystal , Haruka befriends Makoto in order to get close to her . 97 `` Damp Spirits '' `` The Labyrinth of Water : Ami Targeted '' `` Mizu no rabarinsu ! Nerawareta Ami '' ( 水 の ラビリンス ! ねら われ た 亜美 ) Junichi Sato Megumi Sugihara Kazuyuki Hashimoto Katsumi Tamegai May 21 , 1994 June 21 , 2000 Ami 's desire to avoid competition leads her to offending Michiru inadvertently in a swimming race , as well as Usagi accidentally hurting Ami 's feelings . Meanwhile , a Daimon selects Ami as a target . 98 `` Friendly Foes '' `` To Save Our Friends : Moon and Uranus Join Forces '' `` Tomodachi wo sukue ! Mūn Uranusu rengō '' ( 友達 を 救え ! ムーン ウラヌス 連合 ) Yūji Endō Sukehiro Tomita Kenichi Tajiri Taichi Nakamura May 28 , 1994 June 22 , 2000 After Sailor Neptune is captured and possibly killed by Kaolinite , Sailor Moon and Sailor Uranus find themselves shackled together and forced to reassess their less - than - friendly attitudes toward each other in order to save Sailor Neptune from further harm . 99 `` Mixed Emotions '' `` A Man 's Kindness : Yuichiro Heartbroken by Rei '' `` Otoko no yasashisa ! Yūichirō Rei ni shitsuren ? '' ( 男 の 優し さ ! 雄一郎 レイ に 失恋 ? ) Harume Kosaka Megumi Sugihara Yoshiyuki Shikano Shinya Hasegawa June 18 , 1994 June 23 , 2000 Worried by Rei 's intensive meditation , Yuuichirou becomes concerned that she has fallen deeply head - over-heels in love for another man . When he sees her with Haruka and mistakes her for a man , Yuuichirou is convinced that `` he '' is Rei 's new romantic suitor . 100 `` Individual Happiness '' `` I Want to Quit Being a Sailor Guardian : Minako 's Dilemma '' `` Sērā Senshi wo yametai ! ? Minako no nayami '' ( S 戦士 を 辞め たい ! ? 美奈子 の 悩み ) Hiroki Shibata Yoji Enokido Kazuhisa Asai Mari Tominaga June 25 , 1994 June 26 , 2000 Minako is despondent about her double life of being a Sailor Guardian and a normal girl . Meanwhile , an old friend of hers , who she once fell in love with , is attacked by a Daimon , and Minako , as Sailor Venus , must use her skills of volleyball to save him . 101 `` Birthday Blues , Part 1 '' `` Usagi in Tears : a Glass Slipper for My Birthday '' `` Usagi namida ! Tanjōbi ni garasu no kutsu wo '' ( うさぎ 涙 ! 誕生 日 に ガラス の 靴 を ) Yūji Endō Katsuyuki Sumisawa Kazuyuki Hashimoto Masahiro Andō July 2 , 1994 June 27 , 2000 Usagi is furious when Mamoru forgets her 15th birthday . When Mamoru buys her an expensive pair of glass slippers to make up for it , he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg to steal her pure heart crystal . To make matters worse , Kaolinite steals Usagi 's Cosmic Heart Compact leaving her unable to transform , and Tuxedo Mask gets captured . 102 `` Birthday Blues , Part Two '' `` The Stolen Pure Heart : Usagi 's Crisis '' `` Ubawareta pyua na kokoro ! Usagi zettai zetsumei '' ( 奪 われ た 純 な 心 ! うさぎ 絶 体 絶命 ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Taichi Nakamura July 16 , 1994 June 28 , 2000 With the Cosmic Heart Compact stolen and Tuxedo Mask kidnapped , Usagi and the four other Sailor Guardians , along with Sailors Uranus and Neptune , go to retake the compact so Usagi can become Sailor Moon again , rescue Tuxedo Mask and face Kaolinite in a final showdown at Tokyo Tower . 103 `` Hello , Sailor Mini Moon '' `` The Arrival of a Tiny Pretty Guardian '' `` Yatte kita chiccha na Bishōjo Senshi '' ( やって来 た ちっちゃな 美 少女 戦士 ) Kunihiko Ikuhara Shigeru Yanagawa Yoshiyuki Shikano Kazuya Kuroda August 6 , 1994 June 29 , 2000 Eudial , and the Witches 5 are tasked to take over from Kaolinite 's position , in which they target a genius drummer recruited to play at a festival for Rei . The Sailor Guardians can not defeat the Daimon who attacks her , but Chibiusa arrives as the Guardians ' newest ally , Sailor Chibi - Moon . 104 `` Tainted Tea Party '' `` Making New Friends : Chibi Moon 's Adventure '' `` Tomodachi wo motomete ! Chibi Mūn no katsuyaku '' ( 友達 を 求め て ! ちび ムーン の 活躍 ) Junichi Sato Megumi Sugihara Kazuhisa Asai Ikuko Itō August 20 , 1994 June 30 , 2000 Chibiusa , who has returned from Crystal Tokyo for further training , attempts to adapt to her new home by making friends her age . She tries to befriend a master who does tea ceremonies , only to become the Death Busters ' latest target . 105 `` People Who Need People '' `` I Want Power : Mako Lost in Doubt '' `` Pawā ga hoshī ! Mako - chan no mayoi michi '' ( 力 が 欲しい ! まこ ちゃん の 迷い 道 ) Hiroki Shibata Shigeru Yanagawa Yoshiyuki Shikano Mari Tominaga August 27 , 1994 July 3 , 2000 Makoto is training in the mountains , trying to become much stronger after being easily defeated by a Daimon before . Her trainer is attacked , and tells her that nobody can live alone . She attacks the Daimon , fails , and realizes she needs her best friends in order to win . 106 `` Related By Destiny '' `` The Bond of Destiny : Uranus 's Distant Past '' `` Unmei no kizuna ! Uranusu no tōi hi '' ( 運命 の きずな ! ウラヌス の 遠い 日 ) Takuya Igarashi Yoji Enokido Kenichi Tajiri Katsumi Tamegai September 3 , 1994 July 4 , 2000 Haruka remembers the past when she first met Michiru , who told her that she was destined to become Sailor Uranus . Haruka initially refused that destiny , for she loved her life too much and was afraid of change . Haruka was attacked , and Sailor Neptune was injured while defending her . In her anguish , Haruka took the transformation pen and accepted her destiny as Sailor Uranus , the Sailor Guardian of the Sky . 107 `` Art Appreciation '' `` Art Is an Explosion of Love : Chibiusa 's First Love '' `` Geijutsu wa ai no bakuhatsu ! Chibiusa no hatsukoi '' ( 芸術 は 愛 の 爆発 ! ちびうさ の 初恋 ) Yūji Endō Katsuyuki Sumisawa Minoru Ōkōchi Taichi Nakamura September 10 , 1994 July 5 , 2000 Chibiusa meets a boy named Masanori at her art class and develops a crush on him . Unfortunately , he already has a crush on Michiru , who is taking a class at that school as well . When a Daimon attacks Masanori , Chibiusa must take a stand to protect him . 108 `` Everything 's Coming Up Rosey '' `` Usagi Dancing to the Waltz '' `` Usagi no dansu wa warutsu ni notte '' ( うさぎ の ダンス は ワルツ に 乗っ て ) Harume Kosaka Megumi Sugihara Kazuhisa Asai Masahiro Andō September 17 , 1994 July 6 , 2000 An English gentleman named Edwards invites Mamoru and the girls to a fancy party , where Usagi accidentally becomes drunk and the team is forced to confront a fashionable Eudial and her latest Daimon . 109 `` No Turning Back '' `` The Shocking Moment : Everyone 's Identities Revealed '' `` Shōgeki no toki ! Akasareta tagai no shōtai '' ( 衝撃 の 刻 ! 明か され た 互い の 正体 ) Junichi Sato Sukehiro Tomita Yoshiyuki Shikano Kazuya Kuroda September 24 , 1994 July 7 , 2000 Minako becomes jealous that a Daimon has not yet attacked her , and she fears that she does not have a pure heart . After doing as many `` pure '' things as she can , she is attacked , and her pure heart is taken from her . When Eudial nearly gets away , Usagi is forced to reveal her identity to Haruka & Michiru , who in turn , reveal their identities as Sailors Uranus and Neptune to Sailor Moon . 110 `` Destiny 's Arrival '' `` The Death of Uranus and Neptune : The Talisman Appear '' `` Uranusu - tachi no shi ? Tarisuman shutsugen '' ( ウラヌス 達 の 死 ? タリスマン 出現 ) Kunihiko Ikuhara Yoji Enokido Kenichi Tajiri Mari Tominaga October 15 , 1994 July 10 , 2000 Faced with losing her position at Witches 5 , Eudial creates a computer program to find the true owners of the three Talismans , and learns that Haruka and Michiru are the ones who hold two of the Talismans within their own hearts ! They take away Usagi 's brooch , and follow Eudial onto a lonely island . However , the invitation turns out to be a trick , and Michiru 's pure heart is taken , which turns into the mystical Deep Aqua Mirror . Haruka , in order to release her own talisman , shoots herself with Eudial 's gun . Her pure heart transforms into the mystical Space Sword . 111 `` The Purity Chalice '' `` The Holy Grail 's Mystical Power : Moon 's Double Transformation '' `` Seihai no shinpi na chikara ! Mūn nidan henshin '' ( 聖杯 の 神秘 な 力 ! ムーン 二 段 変身 ) Hiroki Shibata Sukehiro Tomita Kazuyuki Hashimoto Ikuko Itō October 22 , 1994 July 11 , 2000 Sailor Pluto appears and reawakens Sailors Uranus and Neptune , and the three talismans resonate with each other ( the third and last being Sailor Pluto 's Garnet Orb ) . The Holy Grail appears , and Sailor Moon grabs it and evolves into Super Sailor Moon , who drives Eudial out . Mimete then kills Eudial by sabotaging her car with snails . 112 `` Show Stoppers '' `` Who is the True Messiah ? Chaos of Light and Darkness '' `` Shin no kyūseishu wa dare ? Hikari to kage no kaosu '' ( 真 の 救世主 は 誰 ? 光 と 影 の カオス ) Noriyo Sasaki Shigeru Yanagawa Minoru Ōkōchi Shinya Hasegawa November 5 , 1994 July 12 , 2000 The girls end up watching the filming of a movie , where Chibiusa encounters Hotaru , whom she quickly befriends . But in the end , she is revealed to be the 12 - year - old daughter of Professor Souichi Tomoe , the possessed leader of the Death Busters . Super Sailor Moon uses her new `` Rainbow Moon Heartache '' attack to defeat the Daimon of the day . 113 `` Rini 's Risky Friendship '' `` A House Filled With Evil Presence : The Beautiful Hotaru 's Secret '' `` Yōki tadayou ie ! Bishōjo Hotaru no himitsu '' ( 妖気 漂う 家 ! 美 少女 ほたる の 秘密 ) Kōnosuke Uda Megumi Sugihara Kenichi Tajiri Taichi Nakamura November 12 , 1994 July 13 , 2000 The three Outer Guardians appear to Tuxedo Mask , and tell him why they can not fight alongside Sailor Moon and the other four Sailor Guardians . Usagi and Chibiusa visit Hotaru , and see a woman who looks like Kaolinite . The Messiah of Silence appears demanding pure hearts . 114 `` Mimet 's Mess '' `` I Love Idols : Mimete 's Dilemma '' `` Aidoru daisuki ! Nayameru Mimetto '' ( アイドル 大好き ! 悩め る ミメット ) Harume Kosaka Yoji Enokido Yoshiyuki Shikano Masahiro Andō November 19 , 1994 July 14 , 2000 Minako and Mimete enter a beauty contest at the same time , with Mimete willing to risk it all to obtain victory . Both fall short of the needed votes to win the idol position , and Mimete decides to just go back to her evil ways . 115 `` The Shadow of Silence '' `` Shadow of Silence : The Pale Glimmer of a Firefly '' `` Chinmoku no kage ! ? Awaki Hotaru hi no yurameki '' ( 沈黙 の 影 ! ? あ わき 蛍火 の ゆらめき ) Takuya Igarashi Yoji Enokido Kenichi Tajiri Katsumi Tamegai November 26 , 1994 July 17 , 2000 Hotaru overexerts herself , and is hospitalized , during which her past without friends is shown , and the reasons for it . Meanwhile , Mimete targets a soap opera actor , during which the attacking Daimon runs near Hotaru , who glows and completely freezes the Daimon . 116 `` Thorny Weather '' `` Sunny Skies After a Storm : A Friendship Dedicated to Hotaru '' `` Arashi nochi hare ! Hotaru ni sasageru yūjō '' ( 嵐 のち 晴れ ! ほたる に 捧げる 友情 ) Yūji Endō Megumi Sugihara Kazuhisa Asai Jouji Yanase December 3 , 1994 July 18 , 2000 Mamoru takes Usagi , Chibiusa and Hotaru to a botanical garden that is tended by Yoshiki Kurebayashi , an old classmate of his . While there , Mimete targets Yoshiki 's pure heart and causes a storm that puts the plants at risk . Sailor Moon defeats the accompanying Daimon , U-Bara . 117 `` Heightened Hazard '' `` Higher and Stronger : A Cheer from Usagi '' `` Yori takaku yori tsuyoku ! Usagi no ōen '' ( より 高く より 強く ! うさぎ の 応援 ) Kunihiko Ikuhara Shigeru Yanagawa Minoru Ōkōchi Taichi Nakamura December 10 , 1994 July 19 , 2000 Hotaru wishes to give a letter to Shun Hayase , a famous athlete . He becomes targeted when she goes to give it to him , and she possessively blasts the enemy with a hugely strong amount of energy . She talks to Shun about being frail and weak , and he thanks her for talking to him . 118 `` It 's In The Cards '' `` The Battle Inside the Demonic Space : The Sailor Guardians ' Gamble '' `` Makū no tatakai ! Sērā Senshi no kake '' ( 魔 空 の 戦い ! セーラー 戦士 の 賭け ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Masahide Yanagisawa December 17 , 1994 July 20 , 2000 Mimete attempts to create a Daimon by herself , throwing the entire Tomoe house into another dimension . The Guardians enter the dimension and are defeated by a game - master Daimon named U-Ikasaman . Hotaru manages to assist in her defeat by the use of a bluff . 119 `` Goodness Eclipsed '' `` The Messiah of Silence Awakens ? Stars of Destiny '' `` Chinmoku no Meshia kakusei ? Unmei no hoshiboshi '' ( 沈黙 の メシア の 覚せい ? 運命 の 星々 ) Harume Kosaka Yoji Enokido Kazuhisa Asai Shinya Hasegawa December 24 , 1994 September 13 , 2000 The Sailor Guardians visit a planetarium , where a man is attacked by a Daimon . The symbol of Saturn appears on Hotaru 's forehead , and she attacks the Daimon . However , Sailor Uranus , Neptune , and Pluto attack Hotaru , since she is both Sailor Saturn , the Sailor Guardian of Silence , Ruin and Birth , and also possessed by the Messiah of Silence . Super Sailor Moon takes the blow to protect Hotaru . 120 `` Next In Line '' `` An Invasion from an Another Dimension : Mystery of Infinity Academy '' `` Ijigen kara no shinryaku ! Mugen Gakuen no nazo '' ( 異 次元 から の 侵略 ! 無限 学園 の 謎 ) Kōnosuke Uda Megumi Sugihara Minoru Ōkōchi Katsumi Tamegai January 7 , 1995 July 21 , 2000 Chibiusa goes to Hotaru 's house to find that Hotaru has disappeared , and is devastated about it . At the same time , the Sailor Guardians investigate Infinity Academy , where Mimete tries one final time to defeat them before Tellu literally pulls the plug on her operation . 121 `` Fiendish Ferns '' `` A Bewitching Flower That Steals Hearts : Tellu , the Third Witch '' `` Kokoro wo ubau yōka ! Daisan no majō Teruru '' ( 心 を 奪う 妖花 ! 第 三 の 魔女 テルル ) Takuya Igarashi Shigeru Yanagawa Kazuyuki Hashimoto Taichi Nakamura January 14 , 1995 July 24 , 2000 Tellu develops a new plan to steal pure hearts , using a plant bred for that purpose . Setsuna realizes it is an alien plant made to steal pure hearts , and warns the Sailor Guardians . Tuxedo Mask causes a mutated plant to destroy Tellu , leading to them tracing her back to the mysterious Infinity Academy . 122 `` The Science Of Love '' `` Believe in Love : Ami , the Kindhearted Guardian '' `` Ai wo shinjite ! Ami , kokoro yasashiki Senshi '' ( 愛 を 信じ て ! 亜美 心 優しき 戦士 ) Yūji Endō Sukehiro Tomita Kenichi Tajiri Hideyuki Motohashi January 21 , 1995 July 25 , 2000 Ami investigates the Infinity Academy , where she meets Viluy , the fourth of the Witches 5 . Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers . But after her battle , her own microscopic robots turn against her , destroying her in the process . 123 `` Wake Up Call '' `` Shadows of Destruction : the Messiah of Silence Awakens '' `` Hametsu no kage ! Chinmoku no Meshia no mezame '' ( 破滅 の 影 ! 沈黙 の メシア の 目覚め ) Harume Kosaka Megumi Sugihara Minoru Ōkōchi Mari Tominaga January 28 , 1995 July 26 , 2000 The last of the Witches 5 , Cyprine steals pure hearts from Infinity Academy students , and the Sailor Guardians engage Cyprine , and her alter - ego , Ptilol , in a showdown . Meanwhile , Chibiusa is abducted by the recently resurrected Kaolinite , and her pure heart is stolen by Mistress 9 , having taken over Hotaru 's body , which allows her to awaken as the Messiah of Silence . 124 `` Who 's Really Who ? '' `` The Coming Terror of Darkness : Struggle of the Eight Guardians '' `` Semari kuru yami no kyōfu ! Kusen no Hassenshi '' ( 迫り 来る 闇 の 恐怖 ! 苦戦 の 8 戦士 ) Takuya Igarashi Yoji Enokido Kazuyuki Hashimoto Ikuko Itō February 4 , 1995 July 28 , 2000 Mistress 9 uses all remaining Daimon eggs to create a powerful energized shield of evil minions around Infinity Academy . Sailor Moon is abducted , as she is the holder of the Holy Grail . After witnessing Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time , Sailor Uranus and Sailor Neptune destroy Germatoid , the Daimon possessing Professor Souichi Tomoe with their talismans , the Deep Aqua Mirror and the Space Sword . 125 `` Darkness , My Old Friend '' `` The Shining Shooting Star : Saturn and the Messiah '' `` Kagayaku ryūsei ! Satān soshite kyūseishu '' ( 輝く 流星 ! サターン そして 救世主 ) Kōnosuke Uda Yoji Enokido Kenichi Tajiri Ikuko Itō February 11 , 1995 July 29 , 2000 Sailor Moon is tricked into giving Mistress 9 the Holy Grail , which she uses to summon Pharaoh 90 to Earth . Hotaru regains consciousness with the assistance of Sailor Moon and her father , destroying Mistress 9 and fully awakening the spirit of Sailor Saturn within herself at last . Sailor Saturn herself returns Chibiusa 's pure heart crystal , and then appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers , but Sailor Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more , and save both Hotaru and Sailor Saturn , who is reborn as a baby immediately afterward . 126 `` Second Chance '' `` A New Life : Parting of the Stars of Destiny '' `` Atarashiki seimei ! Unmei no hoshiboshi betsuri no toki '' ( 新しき 生命 ! 運命 の 星々 別離 の 時 ) Yūji Endō Megumi Sugihara Kazuhisa Asai Taichi Nakamura February 18 , 1995 July 30 , 2000 Haruka and Michiru give the newly reborn infant Hotaru to a still - healing Professor Souichi Tomoe , who is finally free of the Death Busters . However , they viciously attack Sailor Moon , for she protected Hotaru 's body and Sailor Saturn , and risked the whole entire world for one mere life . The Silver Crystal throws them back , and they kneel to her as their leader and leave the city . 127 `` Tough Kindness '' `` A Guardian 's Realization : Strength Lies Within a Pure Heart '' `` Senshi no Jikaku ! Tsuyosa wa Pyua na Kokoro no Naka ni '' ( 戦士 の 自覚 ! 強 さ は 純 な 心 の 中 に ) Harume Kosaka Sukehiro Tomita February 25 , 1995 August 1 , 2000 Chibiusa returns to the future after receiving a letter from her mother , Neo-Queen Serenity . Meanwhile , a final Daimon emerges from the rubble of Infinity Academy , and Chibiusa returns just in time to help defeat it .
  Fact 0.0: The table columns are No., Cloverway Inc. title, original Japanese and Viz Media titles, episode director(s), writer(s), art director(s), animation director(s), original airdate, and English airdate in the United States.
  Fact 0.1: Episode 90 has the Cloverway Inc. title 'Star Struck, Bad Luck', the Viz Media title 'Premonition of the Apocalypse: The Mysterious New Guardians Appear', and the original Japanese title romanized as 'Chikyū hōkai no yokan? Nazo no shin Senshi shutsugen'.
  Fact 0.2: Episode 90 was directed by Junichi Sato, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Masahiro Andō. It originally aired on March 19, 1994 and aired in English in the United States on June 12, 2000.
  Fact 0.3: After experiencing apocalyptic visions, Rei is attacked by a powerful monster called a Daimon.
  Fact 0.4: Sailor Moon is unable to fight the Daimon after it disables her brooch, which reverses her transformation.
  Fact 0.5: The Daimon is defeated by a pair of mysterious new warriors.
  Fact 0.6: The girls despondently realize that a new conflict has begun.
  Fact 0.7: Episode 91 has the Cloverway Inc. title 'Crystal Clear Again', the Viz Media title 'The Rod of Love Is Born: Usagi's New Transformation', and the original Japanese title romanized as 'Ai no roddo tanjō! Usagi no shin henshin'.
  Fact 0.8: Episode 91 was directed by Yūji Endō, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Katsumi Tamegai. It originally aired on March 26, 1994 and aired in English in the United States on June 13, 2000.
  Fact 0.9: While Mamoru strives to cheer Usagi up after her failure in the previous battle, a young girl is the latest target of the Death Busters.
  Fact 0.10: Due to Usagi's inability to transform, the under-powered Sailor Team struggles to fight off the Daimon and is once again forced to rely on the mysterious two warriors.
  Fact 0.11: Once the Sailor Team determines the girl's pure heart is not a talisman, they leave the Sailor Team to fend for themselves.
  Fact 0.12: Tuxedo Mask and Usagi hold hands after the latter's refusal to escape.
  Fact 0.13: Through the power of their love, Usagi's disabled Crystal Star brooch transforms into the Cosmic Heart Compact.
  Fact 0.14: Usagi gains the Spiral Heart Moon Rod and, along with it, the ability to transform and defeat the new enemy.
  Fact 0.15: Episode 92 has the Cloverway Inc. title 'Driving Dangerously', the Viz Media title 'A Handsome Boy? Haruka Tenoh's Secret', and the original Japanese title romanized as 'Suteki na bishōnen? Ten'ō Haruka no himitsu'.
  Fact 0.16: Episode 92 was directed by Kunihiko Ikuhara, written by Katsuyuki Sumisawa, art directed by Kazuhisa Asai, and animation directed by Hisashi Kagawa. It originally aired on April 16, 1994 and aired in English in the United States on June 14, 2000.
  Fact 0.17: Avoiding their study group, Usagi and Minako meet Haruka Tenou, an attractive and supposedly male student of a prestigious local school.
  Fact 0.18: Usagi and Minako both become infatuated with Haruka.
  Fact 0.19: After Minako is defeated in a racing game, Michiru arrives to pick up Haruka.
  Fact 0.20: Michiru and Haruka depart, with Usagi and Minako following behind them soon after.
  Fact 0.21: Once Michiru and Haruka separate at a mechanic's shop, Usagi and Minako take this as their chance.
  Fact 0.22: Haruka tells Usagi and Minako about 'him'-self until a Daimon arrives to attack Kameda, the shop mechanic.
  Fact 0.23: The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down.
  Fact 0.24: Uranus and Neptune arrive to investigate Kameda's pure heart crystal.
  Fact 0.25: When the crystal turns out not to be a talisman, Uranus and Neptune retreat.
  Fact 0.26: Sailor Moon defeats the Daimon.
  Fact 0.27: Later, the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they had told them about.
  Fact 0.28: Michiru arrives with a blonde girl, whom Usagi and Minako initially fail to realize is Haruka.
  Fact 0.29: Once Usagi and Minako realize the blonde girl is Haruka, Minako falls to the ground in anguish.
  Fact 0.30: Haruka notes that she had never said she was male.
  Fact 0.31: Episode 93 has the Cloverway Inc. title 'Bad Harmony', the Viz Media title 'Usagi's Idol: The Graceful Genius Michiru', and the original Japanese title romanized as 'Usagi no akogare! Yūbi no tensai Michiru'.
  Fact 0.32: Episode 93 was directed by Hiroki Shibata, written by Megumi Sugihara, art directed by Kazuyuki Hashimoto, and animation directed by Taichi Nakamura. It originally aired on April 23, 1994 and aired in English in the United States on June 15, 2000.
  Fact 0.33: When Usagi is concerned that she is not smart enough to be with Mamoru after being scolded by the other girls, she seeks advice from Michiru Kaioh.
  Fact 0.34: Michiru Kaioh suggests that Usagi and Mamoru go to a violin recital.
  Fact 0.35: Episode 94 has the Cloverway Inc. title 'Swept Off Her Feet', the Viz Media title 'Protect the Pure Heart: The Three-Way Battle', and the original Japanese title romanized as 'Pyua na kokoro wo mamore! Teki mikata mittsu tomoe ransen'.
  Fact 0.36: Episode 94 was directed by Noriyo Sasaki, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Ikuko Itō. It originally aired on April 30, 1994 and aired in English in the United States on June 16, 2000.
  Fact 0.37: Usagi's friend, Unazuki, is attacked by a Daimon and has her pure heart crystal stolen.
  Fact 0.38: When Sailor Moon tries to take the crystal back to Unazuki, Uranus and Neptune attempt to steal it in order to claim the Talisman.
  Fact 0.39: Episode 95 has the Cloverway Inc. title 'Blinded By Love's Light', the Viz Media title 'Let Moon Help With Your Love Problems', and the original Japanese title romanized as 'Koi no otasuke wa Mūn ni omakase'.
  Fact 0.40: Episode 95 was directed by Harume Kosaka, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Mari Tominaga. It originally aired on May 7, 1994 and aired in English in the United States on June 19, 2000.
  Fact 0.41: Naru and Umino enter an affection contest in order to prove their strong mutual romantic love for each other.
  Fact 0.42: Naru and Umino find themselves targeted by a Daimon.
  Fact 0.43: Episode 96 has the Cloverway Inc. title 'Lita Borrows Trouble', the Viz Media title 'Coldhearted Uranus: Makoto in Danger', and the original Japanese title romanized as 'Reikoku na Uranusu? Makoto no pinchi'.
  Fact 0.44: Episode 96 was directed by Kōnosuke Uda, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Masahiro Andō. It originally aired on May 14, 1994 and aired in English in the United States on June 20, 2000.
  Fact 0.45: Makoto is targeted by a Daimon, but manages to temporarily evade it.
  Fact 0.46: Haruka wants to analyze Makoto's pure heart crystal, and befriends Makoto in order to get close to her.
  Fact 0.47: Episode 97 has the Cloverway Inc. title 'Damp Spirits', the Viz Media title 'The Labyrinth of Water: Ami Targeted', and the original Japanese title romanized as 'Mizu no rabarinsu! Nerawareta Ami'.
  Fact 0.48: Episode 97 was directed by Junichi Sato, written by Megumi Sugihara, art directed by Kazuyuki Hashimoto, and animation directed by Katsumi Tamegai. It originally aired on May 21, 1994 and aired in English in the United States on June 21, 2000.
  Fact 0.49: Ami's desire to avoid competition leads her to inadvertently offend Michiru in a swimming race.
  Fact 0.50: Usagi accidentally hurts Ami's feelings.
  Fact 0.51: Meanwhile, a Daimon selects Ami as a target.
  Fact 0.52: Episode 98 has the Cloverway Inc. title 'Friendly Foes', the Viz Media title 'To Save Our Friends: Moon and Uranus Join Forces', and the original Japanese title romanized as 'Tomodachi wo sukue! Mūn Uranusu rengō'.
  Fact 0.53: Episode 98 was directed by Yūji Endō, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on May 28, 1994 and aired in English in the United States on June 22, 2000.
  Fact 0.54: After Sailor Neptune is captured and possibly killed by Kaolinite, Sailor Moon and Sailor Uranus find themselves shackled together.
  Fact 0.55: Sailor Moon and Sailor Uranus are forced to reassess their less-than-friendly attitudes toward each other in order to save Sailor Neptune from further harm.
  Fact 0.56: Episode 99 has the Cloverway Inc. title 'Mixed Emotions', the Viz Media title 'A Man's Kindness: Yuichiro Heartbroken by Rei', and the original Japanese title romanized as 'Otoko no yasashisa! Yūichirō Rei ni shitsuren?'.
  Fact 0.57: Episode 99 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Yoshiyuki Shikano, and animation directed by Shinya Hasegawa. It originally aired on June 18, 1994 and aired in English in the United States on June 23, 2000.
  Fact 0.58: Worried by Rei's intensive meditation, Yuuichirou becomes concerned that she has fallen deeply head-over-heels in love for another man.
  Fact 0.59: When Yuuichirou sees Rei with Haruka and mistakes Haruka for a man, he is convinced that 'he' is Rei's new romantic suitor.
  Fact 0.60: Episode 100 has the Cloverway Inc. title 'Individual Happiness', the Viz Media title 'I Want to Quit Being a Sailor Guardian: Minako's Dilemma', and the original Japanese title romanized as 'Sērā Senshi wo yametai!? Minako no nayami'.
  Fact 0.61: Episode 100 was directed by Hiroki Shibata, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Mari Tominaga. It originally aired on June 25, 1994 and aired in English in the United States on June 26, 2000.
  Fact 0.62: Minako is despondent about her double life of being a Sailor Guardian and a normal girl.
  Fact 0.63: Meanwhile, an old friend of Minako's, who she once fell in love with, is attacked by a Daimon.
  Fact 0.64: Minako, as Sailor Venus, must use her skills of volleyball to save the old friend.
  Fact 0.65: Episode 101 has the Cloverway Inc. title 'Birthday Blues, Part 1', the Viz Media title 'Usagi in Tears: a Glass Slipper for My Birthday', and the original Japanese title romanized as 'Usagi namida! Tanjōbi ni garasu no kutsu wo'.
  Fact 0.66: Episode 101 was directed by Yūji Endō, written by Katsuyuki Sumisawa, art directed by Kazuyuki Hashimoto, and animation directed by Masahiro Andō. It originally aired on July 2, 1994 and aired in English in the United States on June 27, 2000.
  Fact 0.67: Usagi is furious when Mamoru forgets her 15th birthday.
  Fact 0.68: When Mamoru buys her an expensive pair of glass slippers to make up for it, he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg to steal her pure heart crystal.
  Fact 0.69: Kaolinite steals Usagi's Cosmic Heart Compact, leaving her unable to transform.
  Fact 0.70: Tuxedo Mask gets captured.
  Fact 0.71: Episode 102 has the Cloverway Inc. title 'Birthday Blues, Part Two', the Viz Media title 'The Stolen Pure Heart: Usagi's Crisis', and the original Japanese title romanized as 'Ubawareta pyua na kokoro! Usagi zettai zetsumei'.
  Fact 0.72: Episode 102 was directed by Noriyo Sasaki, written by Katsuyuki Sumisawa, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on July 16, 1994 and aired in English in the United States on June 28, 2000.
  Fact 0.73: With the Cosmic Heart Compact stolen and Tuxedo Mask kidnapped, Usagi and the four other Sailor Guardians, along with Sailors Uranus and Neptune, go to retake the compact.
  Fact 0.74: They go to retake the compact so Usagi can become Sailor Moon again and rescue Tuxedo Mask.
  Fact 0.75: They face Kaolinite in a final showdown at Tokyo Tower.
  Fact 0.76: Episode 103 has the Cloverway Inc. title 'Hello, Sailor Mini Moon', the Viz Media title 'The Arrival of a Tiny Pretty Guardian', and the original Japanese title romanized as 'Yatte kita chiccha na Bishōjo Senshi'.
  Fact 0.77: Episode 103 was directed by Kunihiko Ikuhara, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Kazuya Kuroda. It originally aired on August 6, 1994 and aired in English in the United States on June 29, 2000.
  Fact 0.78: Eudial and the Witches 5 are tasked to take over from Kaolinite's position.
  Fact 0.79: The Witches 5 target a genius drummer recruited to play at a festival for Rei.
  Fact 0.80: The Sailor Guardians cannot defeat the Daimon who attacks the drummer.
  Fact 0.81: Chibiusa arrives as the Guardians' newest ally, Sailor Chibi-Moon.
  Fact 0.82: Episode 104 has the Cloverway Inc. title 'Tainted Tea Party', the Viz Media title 'Making New Friends: Chibi Moon's Adventure', and the original Japanese title romanized as 'Tomodachi wo motomete! Chibi Mūn no katsuyaku'.
  Fact 0.83: Episode 104 was directed by Junichi Sato, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Ikuko Itō. It originally aired on August 20, 1994 and aired in English in the United States on June 30, 2000.
  Fact 0.84: Chibiusa has returned from Crystal Tokyo for further training.
  Fact 0.85: Chibiusa attempts to adapt to her new home by making friends her age.
  Fact 0.86: Chibiusa tries to befriend a master who does tea ceremonies, only to become the Death Busters' latest target.
  Fact 0.87: Episode 105 has the Cloverway Inc. title 'People Who Need People', the Viz Media title 'I Want Power: Mako Lost in Doubt', and the original Japanese title romanized as 'Pawā ga hoshī! Mako-chan no mayoi michi'.
  Fact 0.88: Episode 105 was directed by Hiroki Shibata, written by Shigeru Yanagawa, art directed by Yoshiyuki Shikano, and animation directed by Mari Tominaga. It originally aired on August 27, 1994 and aired in English in the United States on July 3, 2000.
  Fact 0.89: Makoto is training in the mountains, trying to become much stronger after being easily defeated by a Daimon before.
  Fact 0.90: Makoto's trainer is attacked, and tells her that nobody can live alone.
  Fact 0.91: Makoto attacks the Daimon and fails, and realizes she needs her best friends in order to win.
  Fact 0.92: Episode 106 has the Cloverway Inc. title 'Related By Destiny', the Viz Media title 'The Bond of Destiny: Uranus's Distant Past', and the original Japanese title romanized as 'Unmei no kizuna! Uranusu no tōi hi'.
  Fact 0.93: Episode 106 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Katsumi Tamegai. It originally aired on September 3, 1994 and aired in English in the United States on July 4, 2000.
  Fact 0.94: Haruka remembers the past when she first met Michiru, who told her that she was destined to become Sailor Uranus.
  Fact 0.95: Haruka initially refused that destiny, for she loved her life too much and was afraid of change.
  Fact 0.96: Haruka was attacked, and Sailor Neptune was injured while defending her.
  Fact 0.97: In her anguish, Haruka took the transformation pen and accepted her destiny as Sailor Uranus, the Sailor Guardian of the Sky.
  Fact 0.98: Episode 107 has the Cloverway Inc. title 'Art Appreciation', the Viz Media title 'Art Is an Explosion of Love: Chibiusa's First Love', and the original Japanese title romanized as 'Geijutsu wa ai no bakuhatsu! Chibiusa no hatsukoi'.
  Fact 0.99: Episode 107 was directed by Yūji Endō, written by Katsuyuki Sumisawa, art directed by Minoru Ōkōchi, and animation directed by Taichi Nakamura. It originally aired on September 10, 1994 and aired in English in the United States on July 5, 2000.
  Fact 0.100: Chibiusa meets a boy named Masanori at her art class and develops a crush on him.
  Fact 0.101: Masanori already has a crush on Michiru, who is taking a class at that school as well.
  Fact 0.102: When a Daimon attacks Masanori, Chibiusa must take a stand to protect him.
  Fact 0.103: Episode 108 has the Cloverway Inc. title 'Everything's Coming Up Rosey', the Viz Media title 'Usagi Dancing to the Waltz', and the original Japanese title romanized as 'Usagi no dansu wa warutsu ni notte'.
  Fact 0.104: Episode 108 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Masahiro Andō. It originally aired on September 17, 1994 and aired in English in the United States on July 6, 2000.
  Fact 0.105: An English gentleman named Edwards invites Mamoru and the girls to a fancy party.
  Fact 0.106: At the party, Usagi accidentally becomes drunk.
  Fact 0.107: At the party, the team is forced to confront a fashionable Eudial and her latest Daimon.
  Fact 0.108: Episode 109 has the Cloverway Inc. title 'No Turning Back', the Viz Media title 'The Shocking Moment: Everyone's Identities Revealed', and the original Japanese title romanized as 'Shōgeki no toki! Akasareta tagai no shōtai'.
  Fact 0.109: Episode 109 was directed by Junichi Sato, written by Sukehiro Tomita, art directed by Yoshiyuki Shikano, and animation directed by Kazuya Kuroda. It originally aired on September 24, 1994 and aired in English in the United States on July 7, 2000.
  Fact 0.110: Minako becomes jealous that a Daimon has not yet attacked her, and she fears that she does not have a pure heart.
  Fact 0.111: After doing as many 'pure' things as she can, Minako is attacked, and her pure heart is taken from her.
  Fact 0.112: When Eudial nearly gets away, Usagi is forced to reveal her identity to Haruka and Michiru.
  Fact 0.113: Haruka and Michiru reveal their identities as Sailors Uranus and Neptune to Sailor Moon.
  Fact 0.114: Episode 110 has the Cloverway Inc. title 'Destiny's Arrival', the Viz Media title 'The Death of Uranus and Neptune: The Talisman Appear', and the original Japanese title romanized as 'Uranusu-tachi no shi? Tarisuman shutsugen'.
  Fact 0.115: Episode 110 was directed by Kunihiko Ikuhara, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Mari Tominaga. It originally aired on October 15, 1994 and aired in English in the United States on July 10, 2000.
  Fact 0.116: Faced with losing her position at Witches 5, Eudial creates a computer program to find the true owners of the three Talismans.
  Fact 0.117: Eudial learns that Haruka and Michiru are the ones who hold two of the Talismans within their own hearts.
  Fact 0.118: Eudial and her companions take away Usagi's brooch, and follow Eudial onto a lonely island.
  Fact 0.119: The invitation turns out to be a trick, and Michiru's pure heart is taken, which turns into the mystical Deep Aqua Mirror.
  Fact 0.120: Haruka, in order to release her own talisman, shoots herself with Eudial's gun.
  Fact 0.121: Haruka's pure heart transforms into the mystical Space Sword.
  Fact 0.122: Episode 111 has the Cloverway Inc. title 'The Purity Chalice', the Viz Media title 'The Holy Grail's Mystical Power: Moon's Double Transformation', and the original Japanese title romanized as 'Seihai no shinpi na chikara! Mūn nidan henshin'.
  Fact 0.123: Episode 111 was directed by Hiroki Shibata, written by Sukehiro Tomita, art directed by Kazuyuki Hashimoto, and animation directed by Ikuko Itō. It originally aired on October 22, 1994 and aired in English in the United States on July 11, 2000.
  Fact 0.124: Sailor Pluto appears and reawakens Sailors Uranus and Neptune.
  Fact 0.125: The three talismans resonate with each other, the third and last being Sailor Pluto's Garnet Orb.
  Fact 0.126: The Holy Grail appears, and Sailor Moon grabs it and evolves into Super Sailor Moon.
  Fact 0.127: Super Sailor Moon drives Eudial out.
  Fact 0.128: Mimete kills Eudial by sabotaging her car with snails.
  Fact 0.129: Episode 112 has the Cloverway Inc. title 'Show Stoppers', the Viz Media title 'Who is the True Messiah? Chaos of Light and Darkness', and the original Japanese title romanized as 'Shin no kyūseishu wa dare? Hikari to kage no kaosu'.
  Fact 0.130: Episode 112 was directed by Noriyo Sasaki, written by Shigeru Yanagawa, art directed by Minoru Ōkōchi, and animation directed by Shinya Hasegawa. It originally aired on November 5, 1994 and aired in English in the United States on July 12, 2000.
  Fact 0.131: The girls end up watching the filming of a movie, where Chibiusa encounters Hotaru, whom she quickly befriends.
  Fact 0.132: In the end, Chibiusa is revealed to be the 12-year-old daughter of Professor Souichi Tomoe, the possessed leader of the Death Busters.
  Fact 0.133: Super Sailor Moon uses her new 'Rainbow Moon Heartache' attack to defeat the Daimon of the day.
  Fact 0.134: Episode 113 has the Cloverway Inc. title 'Rini's Risky Friendship', the Viz Media title 'A House Filled With Evil Presence: The Beautiful Hotaru's Secret', and the original Japanese title romanized as 'Yōki tadayou ie! Bishōjo Hotaru no himitsu'.
  Fact 0.135: Episode 113 was directed by Kōnosuke Uda, written by Megumi Sugihara, art directed by Kenichi Tajiri, and animation directed by Taichi Nakamura. It originally aired on November 12, 1994 and aired in English in the United States on July 13, 2000.
  Fact 0.136: The three Outer Guardians appear to Tuxedo Mask, and tell him why they cannot fight alongside Sailor Moon and the other four Sailor Guardians.
  Fact 0.137: Usagi and Chibiusa visit Hotaru, and see a woman who looks like Kaolinite.
  Fact 0.138: The Messiah of Silence appears demanding pure hearts.
  Fact 0.139: Episode 114 has the Cloverway Inc. title 'Mimet's Mess', the Viz Media title 'I Love Idols: Mimete's Dilemma', and the original Japanese title romanized as 'Aidoru daisuki! Nayameru Mimetto'.
  Fact 0.140: Episode 114 was directed by Harume Kosaka, written by Yoji Enokido, art directed by Yoshiyuki Shikano, and animation directed by Masahiro Andō. It originally aired on November 19, 1994 and aired in English in the United States on July 14, 2000.
  Fact 0.141: Minako and Mimete enter a beauty contest at the same time, with Mimete willing to risk it all to obtain victory.
  Fact 0.142: Both Minako and Mimete fall short of the needed votes to win the idol position.
  Fact 0.143: Mimete decides to go back to her evil ways.
  Fact 0.144: Episode 115 has the Cloverway Inc. title 'The Shadow of Silence', the Viz Media title 'Shadow of Silence: The Pale Glimmer of a Firefly', and the original Japanese title romanized as 'Chinmoku no kage!? Awaki Hotaru hi no yurameki'.
  Fact 0.145: Episode 115 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Katsumi Tamegai. It originally aired on November 26, 1994 and aired in English in the United States on July 17, 2000.
  Fact 0.146: Hotaru overexerts herself and is hospitalized, during which her past without friends is shown, and the reasons for it.
  Fact 0.147: Meanwhile, Mimete targets a soap opera actor.
  Fact 0.148: During the attack on the actor, the attacking Daimon runs near Hotaru, who glows and completely freezes the Daimon.
  Fact 0.149: Episode 116 has the Cloverway Inc. title 'Thorny Weather', the Viz Media title 'Sunny Skies After a Storm: A Friendship Dedicated to Hotaru', and the original Japanese title romanized as 'Arashi nochi hare! Hotaru ni sasageru yūjō'.
  Fact 0.150: Episode 116 was directed by Yūji Endō, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Jouji Yanase. It originally aired on December 3, 1994 and aired in English in the United States on July 18, 2000.
  Fact 0.151: Mamoru takes Usagi, Chibiusa and Hotaru to a botanical garden that is tended by Yoshiki Kurebayashi, an old classmate of Mamoru's.
  Fact 0.152: While there, Mimete targets Yoshiki's pure heart and causes a storm that puts the plants at risk.
  Fact 0.153: Sailor Moon defeats the accompanying Daimon, U-Bara.
  Fact 0.154: Episode 117 has the Cloverway Inc. title 'Heightened Hazard', the Viz Media title 'Higher and Stronger: A Cheer from Usagi', and the original Japanese title romanized as 'Yori takaku yori tsuyoku! Usagi no ōen'.
  Fact 0.155: Episode 117 was directed by Kunihiko Ikuhara, written by Shigeru Yanagawa, art directed by Minoru Ōkōchi, and animation directed by Taichi Nakamura. It originally aired on December 10, 1994 and aired in English in the United States on July 19, 2000.
  Fact 0.156: Hotaru wishes to give a letter to Shun Hayase, a famous athlete.
  Fact 0.157: Shun Hayase becomes targeted when Hotaru goes to give the letter to him.
  Fact 0.158: Hotaru possessively blasts the enemy with a hugely strong amount of energy.
  Fact 0.159: Hotaru talks to Shun about being frail and weak, and he thanks her for talking to him.
  Fact 0.160: Episode 118 has the Cloverway Inc. title 'It's In The Cards', the Viz Media title 'The Battle Inside the Demonic Space: The Sailor Guardians' Gamble', and the original Japanese title romanized as 'Makū no tatakai! Sērā Senshi no kake'.
  Fact 0.161: Episode 118 was directed by Noriyo Sasaki, written by Katsuyuki Sumisawa, art directed by Kenichi Tajiri, and animation directed by Masahide Yanagisawa. It originally aired on December 17, 1994 and aired in English in the United States on July 20, 2000.
  Fact 0.162: Mimete attempts to create a Daimon by herself, throwing the entire Tomoe house into another dimension.
  Fact 0.163: The Guardians enter the dimension and are defeated by a game-master Daimon named U-Ikasaman.
  Fact 0.164: Hotaru manages to assist in U-Ikasaman's defeat by the use of a bluff.
  Fact 0.165: Episode 119 has the Cloverway Inc. title 'Goodness Eclipsed', the Viz Media title 'The Messiah of Silence Awakens? Stars of Destiny', and the original Japanese title romanized as 'Chinmoku no Meshia kakusei? Unmei no hoshiboshi'.
  Fact 0.166: Episode 119 was directed by Harume Kosaka, written by Yoji Enokido, art directed by Kazuhisa Asai, and animation directed by Shinya Hasegawa. It originally aired on December 24, 1994 and aired in English in the United States on September 13, 2000.
  Fact 0.167: The Sailor Guardians visit a planetarium, where a man is attacked by a Daimon.
  Fact 0.168: The symbol of Saturn appears on Hotaru's forehead, and she attacks the Daimon.
  Fact 0.169: Sailor Uranus, Neptune, and Pluto attack Hotaru, since she is both Sailor Saturn, the Sailor Guardian of Silence, Ruin and Birth, and also possessed by the Messiah of Silence.
  Fact 0.170: Super Sailor Moon takes the blow to protect Hotaru.
  Fact 0.171: Episode 120 has the Cloverway Inc. title 'Next In Line', the Viz Media title 'An Invasion from an Another Dimension: Mystery of Infinity Academy', and the original Japanese title romanized as 'Ijigen kara no shinryaku! Mugen Gakuen no nazo'.
  Fact 0.172: Episode 120 was directed by Kōnosuke Uda, written by Megumi Sugihara, art directed by Minoru Ōkōchi, and animation directed by Katsumi Tamegai. It originally aired on January 7, 1995 and aired in English in the United States on July 21, 2000.
  Fact 0.173: Chibiusa goes to Hotaru's house to find that Hotaru has disappeared, and is devastated about it.
  Fact 0.174: At the same time, the Sailor Guardians investigate Infinity Academy.
  Fact 0.175: Mimete tries one final time to defeat the Sailor Guardians before Tellu literally pulls the plug on her operation.
  Fact 0.176: Episode 121 has the Cloverway Inc. title 'Fiendish Ferns', the Viz Media title 'A Bewitching Flower That Steals Hearts: Tellu, the Third Witch', and the original Japanese title romanized as 'Kokoro wo ubau yōka! Daisan no majō Teruru'.
  Fact 0.177: Episode 121 was directed by Takuya Igarashi, written by Shigeru Yanagawa, art directed by Kazuyuki Hashimoto, and animation directed by Taichi Nakamura. It originally aired on January 14, 1995 and aired in English in the United States on July 24, 2000.
  Fact 0.178: Tellu develops a new plan to steal pure hearts, using a plant bred for that purpose.
  Fact 0.179: Setsuna realizes the plant is an alien plant made to steal pure hearts, and warns the Sailor Guardians.
  Fact 0.180: Tuxedo Mask causes a mutated plant to destroy Tellu.
  Fact 0.181: This leads the Sailor Guardians to trace Tellu back to the mysterious Infinity Academy.
  Fact 0.182: Episode 122 has the Cloverway Inc. title 'The Science Of Love', the Viz Media title 'Believe in Love: Ami, the Kindhearted Guardian', and the original Japanese title romanized as 'Ai wo shinjite! Ami, kokoro yasashiki Senshi'.
  Fact 0.183: Episode 122 was directed by Yūji Endō, written by Sukehiro Tomita, art directed by Kenichi Tajiri, and animation directed by Hideyuki Motohashi. It originally aired on January 21, 1995 and aired in English in the United States on July 25, 2000.
  Fact 0.184: Ami investigates the Infinity Academy, where she meets Viluy, the fourth of the Witches 5.
  Fact 0.185: Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers.
  Fact 0.186: After Viluy's battle, her own microscopic robots turn against her, destroying her in the process.
  Fact 0.187: Episode 123 has the Cloverway Inc. title 'Wake Up Call', the Viz Media title 'Shadows of Destruction: the Messiah of Silence Awakens', and the original Japanese title romanized as 'Hametsu no kage! Chinmoku no Meshia no mezame'.
  Fact 0.188: Episode 123 was directed by Harume Kosaka, written by Megumi Sugihara, art directed by Minoru Ōkōchi, and animation directed by Mari Tominaga. It originally aired on January 28, 1995 and aired in English in the United States on July 26, 2000.
  Fact 0.189: Cyprine is the last of the Witches 5.
  Fact 0.190: Cyprine steals pure hearts from Infinity Academy students.
  Fact 0.191: The Sailor Guardians engage Cyprine, and her alter-ego, Ptilol, in a showdown.
  Fact 0.192: Chibiusa is abducted by the recently resurrected Kaolinite.
  Fact 0.193: Chibiusa's pure heart is stolen by Mistress 9, who has taken over Hotaru's body, which allows her to awaken as the Messiah of Silence.
  Fact 0.194: Episode 124 has the Cloverway Inc. title 'Who's Really Who?', the Viz Media title 'The Coming Terror of Darkness: Struggle of the Eight Guardians', and the original Japanese title romanized as 'Semari kuru yami no kyōfu! Kusen no Hassenshi'.
  Fact 0.195: Episode 124 was directed by Takuya Igarashi, written by Yoji Enokido, art directed by Kazuyuki Hashimoto, and animation directed by Ikuko Itō. It originally aired on February 4, 1995 and aired in English in the United States on July 28, 2000.
  Fact 0.196: Mistress 9 uses all remaining Daimon eggs to create a powerful energized shield of evil minions around Infinity Academy.
  Fact 0.197: Sailor Moon is abducted, as she is the holder of the Holy Grail.
  Fact 0.198: After witnessing Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time, Sailor Uranus and Sailor Neptune destroy Germatoid.
  Fact 0.199: Germatoid is the Daimon possessing Professor Souichi Tomoe.
  Fact 0.200: Sailor Uranus and Sailor Neptune destroy Germatoid with their talismans, the Deep Aqua Mirror and the Space Sword.
  Fact 0.201: Episode 125 has the Cloverway Inc. title 'Darkness, My Old Friend', the Viz Media title 'The Shining Shooting Star: Saturn and the Messiah', and the original Japanese title romanized as 'Kagayaku ryūsei! Satān soshite kyūseishu'.
  Fact 0.202: Episode 125 was directed by Kōnosuke Uda, written by Yoji Enokido, art directed by Kenichi Tajiri, and animation directed by Ikuko Itō. It originally aired on February 11, 1995 and aired in English in the United States on July 29, 2000.
  Fact 0.203: Sailor Moon is tricked into giving Mistress 9 the Holy Grail, which she uses to summon Pharaoh 90 to Earth.
  Fact 0.204: Hotaru regains consciousness with the assistance of Sailor Moon and her father, destroying Mistress 9.
  Fact 0.205: Hotaru fully awakens the spirit of Sailor Saturn within herself at last.
  Fact 0.206: Sailor Saturn returns Chibiusa's pure heart crystal.
  Fact 0.207: Sailor Saturn appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers.
  Fact 0.208: Sailor Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more.
  Fact 0.209: Super Sailor Moon saves both Hotaru and Sailor Saturn.
  Fact 0.210: Sailor Saturn is reborn as a baby immediately afterward.
  Fact 0.211: Episode 126 has the Cloverway Inc. title 'Second Chance', the Viz Media title 'A New Life: Parting of the Stars of Destiny', and the original Japanese title romanized as 'Atarashiki seimei! Unmei no hoshiboshi betsuri no toki'.
  Fact 0.212: Episode 126 was directed by Yūji Endō, written by Megumi Sugihara, art directed by Kazuhisa Asai, and animation directed by Taichi Nakamura. It originally aired on February 18, 1995 and aired in English in the United States on July 30, 2000.
  Fact 0.213: Haruka and Michiru give the newly reborn infant Hotaru to a still-healing Professor Souichi Tomoe, who is finally free of the Death Busters.
  Fact 0.214: Haruka and Michiru viciously attack Sailor Moon, for she protected Hotaru's body and Sailor Saturn, and risked the whole entire world for one mere life.
  Fact 0.215: The Silver Crystal throws Haruka and Michiru back.
  Fact 0.216: Haruka and Michiru kneel to Sailor Moon as their leader and leave the city.
  Fact 0.217: Episode 127 has the Cloverway Inc. title 'Tough Kindness', the Viz Media title 'A Guardian's Realization: Strength Lies Within a Pure Heart', and the original Japanese title romanized as 'Senshi no Jikaku! Tsuyosa wa Pyua na Kokoro no Naka ni'.
  Fact 0.218: Episode 127 was directed by Harume Kosaka and written by Sukehiro Tomita. It originally aired on February 25, 1995 and aired in English in the United States on August 1, 2000.
  Fact 0.219: Chibiusa returns to the future after receiving a letter from her mother, Neo-Queen Serenity.
  Fact 0.220: Meanwhile, a final Daimon emerges from the rubble of Infinity Academy.
  Fact 0.221: Chibiusa returns just in time to help defeat the final Daimon.

Retrieved items:
<item 1>
List of Sailor Moon S episodes
Satān soshite kyūseishu '' ( 輝く 流星 ! サターン そして 救世主 ) Kōnosuke Uda Yoji Enokido Kenichi Tajiri Ikuko Itō February 11 , 1995 July 29 , 2000 Sailor Moon is tricked
into giving Mistress 9 the Holy Grail , which she uses to summon Pharaoh 90 to Earth . Hotaru regains consciousness with the assistance of Sailor Moon and her
father , destroying Mistress 9 and fully awakening the spirit of Sailor Saturn within herself at last . Sailor Saturn herself returns Chibiusa 's pure heart
crystal , and then appears before Sailor Moon to ultimately sacrifice herself and destroy Pharaoh 90 once and for all with her destructive powers , but Sailor
Moon gains the help of the other six Sailor Guardians to become Super Sailor Moon once more , and save both Hotaru and Sailor Saturn , who is reborn as a baby
immediately afterward . 126 `` Second Chance '' `` A New Life : Parting of the Stars of Destiny '' `` Atarashiki seimei !
</item 1>

<item 2>
List of Sailor Moon S episodes
Meanwhile , Tuxedo Mask and Usagi hold hands after the latter 's refusal to escape , and through the power of their love , Usagi 's disabled Crystal Star brooch
transforms into the Cosmic Heart Compact , and she gains the Spiral Heart Moon Rod and , along with it , the ability to transform and defeat the new enemy . 92
`` Driving Dangerously '' `` A Handsome Boy ? Haruka Tenoh 's Secret '' `` Suteki na bishōnen ? Ten'ō Haruka no himitsu '' ( 素敵 な 美少年 ? 天王 はるか の 秘密 ) Kunihiko
Ikuhara Katsuyuki Sumisawa Kazuhisa Asai Hisashi Kagawa April 16 , 1994 June 14 , 2000 Avoiding their study group , Usagi and Minako meet Haruka Tenou , an
attractive and supposedly male student of a prestigious local school , with whom they both become infatuated .
</item 2>

<item 3>
List of Sailor Moon S episodes
明か され た 互い の 正体 ) Junichi Sato Sukehiro Tomita Yoshiyuki Shikano Kazuya Kuroda September 24 , 1994 July 7 , 2000 Minako becomes jealous that a Daimon has not
yet attacked her , and she fears that she does not have a pure heart . After doing as many `` pure '' things as she can , she is attacked , and her pure heart
is taken from her . When Eudial nearly gets away , Usagi is forced to reveal her identity to Haruka & Michiru , who in turn , reveal their identities as Sailors
Uranus and Neptune to Sailor Moon . 110 `` Destiny 's Arrival '' `` The Death of Uranus and Neptune : The Talisman Appear '' `` Uranusu - tachi no shi ?
Tarisuman shutsugen '' ( ウラヌス 達 の 死 ?
</item 3>

<item 4>
List of Sailor Moon S episodes
After Minako is defeated in a racing game , Michiru arrives to pick up Haruka , and the two depart with Usagi and Minako soon following behind them . Once
Michiru and Haruka separate at a mechanic 's shop , Usagi and Minako take this as their chance , and Haruka tells them about `` him '' - self until a Daimon
arrives to attack Kameda , the shop mechanic . The Daimon starts to escape when Usagi and Minako transform into their Guardian forms to chase it down , when
Uranus and Neptune arrive to investigate Kameda 's pure heart crystal . When it turns out not to be a talisman they retreat and Sailor Moon defeats the Daimon .
Later , the rest of the girls are at Crown Game Center with Usagi and Minako in hopes of seeing the cute boy they 'd told them about ; Michiru arrives with a
blonde girl , whom Usagi and Minako initially fail to realize is Haruka ; once they do , Minako falls to the ground in anguish , and Haruka notes that she 'd
never said she was male .
</item 4>

<item 5>
List of Sailor Moon S episodes
Meanwhile , an old friend of hers , who she once fell in love with , is attacked by a Daimon , and Minako , as Sailor Venus , must use her skills of volleyball
to save him . 101 `` Birthday Blues , Part 1 '' `` Usagi in Tears : a Glass Slipper for My Birthday '' `` Usagi namida ! Tanjōbi ni garasu no kutsu wo '' ( うさぎ
涙 ! 誕生 日 に ガラス の 靴 を ) Yūji Endō Katsuyuki Sumisawa Kazuyuki Hashimoto Masahiro Andō July 2 , 1994 June 27 , 2000 Usagi is furious when Mamoru forgets her 15th
birthday . When Mamoru buys her an expensive pair of glass slippers to make up for it , he fails to realize that Kaolinite sabotaged the shoes with a Daimon egg
to steal her pure heart crystal . To make matters worse , Kaolinite steals Usagi 's Cosmic Heart Compact leaving her unable to transform , and Tuxedo Mask gets
captured .
</item 5>

<item 6>
List of Sailor Moon S episodes
Viluy attempts to use her nanorobots to steal pure hearts from entrance exam takers . But after her battle , her own microscopic robots turn against her ,
destroying her in the process . 123 `` Wake Up Call '' `` Shadows of Destruction : the Messiah of Silence Awakens '' `` Hametsu no kage ! Chinmoku no Meshia no
mezame '' ( 破滅 の 影 ! 沈黙 の メシア の 目覚め ) Harume Kosaka Megumi Sugihara Minoru Ōkōchi Mari Tominaga January 28 , 1995 July 26 , 2000 The last of the Witches 5 ,
Cyprine steals pure hearts from Infinity Academy students , and the Sailor Guardians engage Cyprine , and her alter - ego , Ptilol , in a showdown . Meanwhile ,
Chibiusa is abducted by the recently resurrected Kaolinite , and her pure heart is stolen by Mistress 9 , having taken over Hotaru 's body , which allows her to
awaken as the Messiah of Silence . 124 `` Who 's Really Who ? ''
</item 6>

<item 7>
List of Sailor Moon S episodes
The Daimon is defeated by a pair of mysterious new warriors , leaving the girls to despondently realize that a new conflict has begun . 91 `` Crystal Clear
Again '' `` The Rod of Love Is Born : Usagi 's New Transformation '' `` Ai no roddo tanjō ! Usagi no shin henshin '' ( 愛 の ロッド 誕生 ! うさぎ の 新 変身 ) Yūji Endō
Shigeru Yanagawa Yoshiyuki Shikano Katsumi Tamegai March 26 , 1994 June 13 , 2000 While Mamoru strives to cheer Usagi up after her failure in the previous
battle , a young girl is the latest target of the Death Busters . Due to Usagi 's inability to transform , the under - powered Sailor Team struggles to fight
off the Daimon and are once again forced to rely on the mysterious two warriors . Once they determine the girl 's pure heart is not a talisman , they leave the
Sailor Team to fend for themselves .
</item 7>

<item 8>
List of Sailor Moon S episodes
124 `` Who 's Really Who ? '' `` The Coming Terror of Darkness : Struggle of the Eight Guardians '' `` Semari kuru yami no kyōfu ! Kusen no Hassenshi '' ( 迫り 来る
闇 の 恐怖 ! 苦戦 の 8 戦士 ) Takuya Igarashi Yoji Enokido Kazuyuki Hashimoto Ikuko Itō February 4 , 1995 July 28 , 2000 Mistress 9 uses all remaining Daimon eggs to
create a powerful energized shield of evil minions around Infinity Academy . Sailor Moon is abducted , as she is the holder of the Holy Grail . After witnessing
Sailor Pluto ultimately sacrificing herself to save them by using her forbidden ability to stop time , Sailor Uranus and Sailor Neptune destroy Germatoid , the
Daimon possessing Professor Souichi Tomoe with their talismans , the Deep Aqua Mirror and the Space Sword . 125 `` Darkness , My Old Friend '' `` The Shining
Shooting Star : Saturn and the Messiah '' `` Kagayaku ryūsei !
</item 8>

<item 9>
List of Sailor Moon S episodes
He becomes targeted when she goes to give it to him , and she possessively blasts the enemy with a hugely strong amount of energy . She talks to Shun about
being frail and weak , and he thanks her for talking to him . 118 `` It 's In The Cards '' `` The Battle Inside the Demonic Space : The Sailor Guardians '
Gamble '' `` Makū no tatakai ! Sērā Senshi no kake '' ( 魔 空 の 戦い ! セーラー 戦士 の 賭け ) Noriyo Sasaki Katsuyuki Sumisawa Kenichi Tajiri Masahide Yanagisawa December
17 , 1994 July 20 , 2000 Mimete attempts to create a Daimon by herself , throwing the entire Tomoe house into another dimension . The Guardians enter the
dimension and are defeated by a game - master Daimon named U-Ikasaman . Hotaru manages to assist in her defeat by the use of a bluff . 119 `` Goodness Eclipsed
'' `` The Messiah of Silence Awakens ?
</item 9>

<item 10>
List of Sailor Moon S episodes
タリスマン 出現 ) Kunihiko Ikuhara Yoji Enokido Kenichi Tajiri Mari Tominaga October 15 , 1994 July 10 , 2000 Faced with losing her position at Witches 5 , Eudial
creates a computer program to find the true owners of the three Talismans , and learns that Haruka and Michiru are the ones who hold two of the Talismans within
their own hearts ! They take away Usagi 's brooch , and follow Eudial onto a lonely island . However , the invitation turns out to be a trick , and Michiru 's
pure heart is taken , which turns into the mystical Deep Aqua Mirror . Haruka , in order to release her own talisman , shoots herself with Eudial 's gun . Her
pure heart transforms into the mystical Space Sword . 111 `` The Purity Chalice '' `` The Holy Grail 's Mystical Power : Moon 's Double Transformation '' ``
Seihai no shinpi na chikara !
</item 10>
=== END ITEM id=dd335896c1730d62 ===

=== ITEM id=4cf251de90e9c8b4 ===
Gold sentences and facts:
Sentence 0: Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24 Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5 Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia : 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1
  Fact 0.0: Spain is listed with the value 430.
  Fact 0.1: France is listed with the value 128.
  Fact 0.2: Italy is listed with the value 104.
  Fact 0.3: Russia is listed with the value 100.
  Fact 0.4: Portugal is listed with the value 87.
  Fact 0.5: Germany is listed with the value 77.
  Fact 0.6: United Kingdom is listed with the value 64.
  Fact 0.7: Poland is listed with the value 47.
  Fact 0.8: Greece is listed with the value 46.
  Fact 0.9: Turkey is listed with the value 41.
  Fact 0.10: Belgium is listed with the value 31.
  Fact 0.11: Netherlands is listed with the value 28.
  Fact 0.12: Romania is listed with the value 24.
  Fact 0.13: Switzerland is listed with the value 20.
  Fact 0.14: Austria is listed with the value 14.
  Fact 0.15: Sweden is listed with the value 12.
  Fact 0.16: Croatia is listed with the value 10.
  Fact 0.17: Ireland is listed with the value 9.
  Fact 0.18: Ukraine is listed with the value 9.
  Fact 0.19: Hungary is listed with the value 8.
  Fact 0.20: Bulgaria is listed with the value 6.
  Fact 0.21: Cyprus is listed with the value 6.
  Fact 0.22: Czech Republic is listed with the value 6.
  Fact 0.23: Finland is listed with the value 5.
  Fact 0.24: Lithuania is listed with the value 5.
  Fact 0.25: Norway is listed with the value 5.
  Fact 0.26: Serbia is listed with the value 5.
  Fact 0.27: Slovenia is listed with the value 5.
  Fact 0.28: Denmark is listed with the value 4.
  Fact 0.29: Latvia is listed with the value 4.
  Fact 0.30: Estonia is listed with the value 3.
  Fact 0.31: Azerbaijan is listed with the value 3.
  Fact 0.32: Bosnia and Herzegovina is listed with the value 3.
  Fact 0.33: Georgia is listed with the value 3.
  Fact 0.34: Luxembourg is listed with the value 3.
  Fact 0.35: Slovakia is listed with the value 3.
  Fact 0.36: Armenia is listed with the value 2.
  Fact 0.37: Albania is listed with the value 1.
  Fact 0.38: Andorra is listed with the value 1.
  Fact 0.39: Belarus is listed with the value 1.
  Fact 0.40: Iceland is listed with the value 1.
  Fact 0.41: Macedonia is listed with the value 1.
  Fact 0.42: Malta is listed with the value 1.
  Fact 0.43: Monaco is listed with the value 1.
  Fact 0.44: Montenegro is listed with the value 1.
Sentence 1: United Kingdom : 64
  Fact 1.0: United Kingdom is listed with the value 64.
Sentence 2: Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32 Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20 Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait : 5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24 Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5 Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia : 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1
  Fact 2.0: South Africa is listed under Africa with the value 9.
  Fact 2.1: Egypt is listed under Africa with the value 5.
  Fact 2.2: Morocco is listed under Africa with the value 4.
  Fact 2.3: Tunisia is listed under Africa with the value 3.
  Fact 2.4: Algeria is listed under Africa with the value 1.
  Fact 2.5: United States is listed under Americas with the value 85.
  Fact 2.6: Mexico is listed under Americas with the value 84.
  Fact 2.7: Brazil is listed under Americas with the value 56.
  Fact 2.8: Canada is listed under Americas with the value 32.
  Fact 2.9: Colombia is listed under Americas with the value 14.
  Fact 2.10: Argentina is listed under Americas with the value 10.
  Fact 2.11: Chile is listed under Americas with the value 9.
  Fact 2.12: Venezuela is listed under Americas with the value 8.
  Fact 2.13: Peru is listed under Americas with the value 4.
  Fact 2.14: Puerto Rico is listed under Americas with the value 4.
  Fact 2.15: Dominican Republic is listed under Americas with the value 3.
  Fact 2.16: Guatemala is listed under Americas with the value 3.
  Fact 2.17: Panama is listed under Americas with the value 3.
  Fact 2.18: Costa Rica is listed under Americas with the value 2.
  Fact 2.19: Ecuador is listed under Americas with the value 2.
  Fact 2.20: El Salvador is listed under Americas with the value 2.
  Fact 2.21: Honduras is listed under Americas with the value 2.
  Fact 2.22: Uruguay is listed under Americas with the value 2.
  Fact 2.23: Aruba is listed under Americas with the value 1.
  Fact 2.24: Nicaragua is listed under Americas with the value 1.
  Fact 2.25: Paraguay is listed under Americas with the value 1.
  Fact 2.26: China is listed under Asia with the value 200.
  Fact 2.27: Japan is listed under Asia with the value 99.
  Fact 2.28: South Korea is listed under Asia with the value 43.
  Fact 2.29: Saudi Arabia is listed under Asia with the value 39.
  Fact 2.30: Israel is listed under Asia with the value 24.
  Fact 2.31: India is listed under Asia with the value 20.
  Fact 2.32: Indonesia is listed under Asia with the value 17.
  Fact 2.33: Hong Kong is listed under Asia with the value 14.
  Fact 2.34: Thailand is listed under Asia with the value 11.
  Fact 2.35: United Arab Emirates is listed under Asia with the value 11.
  Fact 2.36: Malaysia is listed under Asia with the value 10.
  Fact 2.37: Singapore is listed under Asia with the value 9.
  Fact 2.38: Taiwan is listed under Asia with the value 9.
  Fact 2.39: Philippines is listed under Asia with the value 8.
  Fact 2.40: Lebanon is listed under Asia with the value 7.
  Fact 2.41: Kazakhstan is listed under Asia with the value 5.
  Fact 2.42: Kuwait is listed under Asia with the value 5.
  Fact 2.43: Qatar is listed under Asia with the value 5.
  Fact 2.44: Jordan is listed under Asia with the value 3.
  Fact 2.45: Bahrain is listed under Asia with the value 2.
  Fact 2.46: Macau is listed under Asia with the value 2.
  Fact 2.47: Oman is listed under Asia with the value 1.
  Fact 2.48: Vietnam is listed under Asia with the value 1.
  Fact 2.49: Spain is listed under Europe with the value 430.
  Fact 2.50: France is listed under Europe with the value 128.
  Fact 2.51: Italy is listed under Europe with the value 104.
  Fact 2.52: Russia is listed under Europe with the value 100.
  Fact 2.53: Portugal is listed under Europe with the value 87.
  Fact 2.54: Germany is listed under Europe with the value 77.
  Fact 2.55: United Kingdom is listed under Europe with the value 64.
  Fact 2.56: Poland is listed under Europe with the value 47.
  Fact 2.57: Greece is listed under Europe with the value 46.
  Fact 2.58: Turkey is listed under Europe with the value 41.
  Fact 2.59: Belgium is listed under Europe with the value 31.
  Fact 2.60: Netherlands is listed under Europe with the value 28.
  Fact 2.61: Romania is listed under Europe with the value 24.
  Fact 2.62: Switzerland is listed under Europe with the value 20.
  Fact 2.63: Austria is listed under Europe with the value 14.
  Fact 2.64: Sweden is listed under Europe with the value 12.
  Fact 2.65: Croatia is listed under Europe with the value 10.
  Fact 2.66: Ireland is listed under Europe with the value 9.
  Fact 2.67: Ukraine is listed under Europe with the value 9.
  Fact 2.68: Hungary is listed under Europe with the value 8.
  Fact 2.69: Bulgaria is listed under Europe with the value 6.
  Fact 2.70: Cyprus is listed under Europe with the value 6.
  Fact 2.71: Czech Republic is listed under Europe with the value 6.
  Fact 2.72: Finland is listed under Europe with the value 5.
  Fact 2.73: Lithuania is listed under Europe with the value 5.
  Fact 2.74: Norway is listed under Europe with the value 5.
  Fact 2.75: Serbia is listed under Europe with the value 5.
  Fact 2.76: Slovenia is listed under Europe with the value 5.
  Fact 2.77: Denmark is listed under Europe with the value 4.
  Fact 2.78: Latvia is listed under Europe with the value 4.
  Fact 2.79: Estonia is listed under Europe with the value 3.
  Fact 2.80: Azerbaijan is listed under Europe with the value 3.
  Fact 2.81: Bosnia and Herzegovina is listed under Europe with the value 3.
  Fact 2.82: Georgia is listed under Europe with the value 3.
  Fact 2.83: Luxembourg is listed under Europe with the value 3.
  Fact 2.84: Slovakia is listed under Europe with the value 3.
  Fact 2.85: Armenia is listed under Europe with the value 2.
  Fact 2.86: Albania is listed under Europe with the value 1.
  Fact 2.87: Andorra is listed under Europe with the value 1.
  Fact 2.88: Belarus is listed under Europe with the value 1.
  Fact 2.89: Iceland is listed under Europe with the value 1.
  Fact 2.90: Macedonia is listed under Europe with the value 1.
  Fact 2.91: Malta is listed under Europe with the value 1.
  Fact 2.92: Monaco is listed under Europe with the value 1.
  Fact 2.93: Montenegro is listed under Europe with the value 1.
  Fact 2.94: Australia is listed under Oceania with the value 19.
  Fact 2.95: New Zealand is listed under Oceania with the value 1.

Retrieved items:
<item 1>
Zara (retailer)
Zara ( retailer ) - Wikipedia Zara ( retailer ) Jump to : navigation , search Zara España , S.A. Type Sociedad Anónima Industry Retail Founded Arteixo , Galicia
( 24 May 1974 ; 43 years ago ( 1974 - 05 - 24 ) ) Founder Amancio Ortega Headquarters Arteixo , Galicia Number of locations 2266 stores Area served Worldwide
Products Clothing Revenue US $ 15.9 billion ( 2016 ) Parent Inditex Website zara.com Zara ( Spanish : ( ˈθaɾa ) ) is a Galician fast fashion ( clothing and
accessories ) retailer based in Arteixo , Galicia . The company was founded in 1975 by Amancio Ortega and Rosalía Mera . It is the main brand of the Inditex
group , the world 's largest apparel retailer . The fashion group also owns brands such as Massimo Dutti , Pull&Bear , Bershka , Stradivarius , Oysho , Zara
Home , and Uterqüe .
</item 1>

<item 2>
Zara (retailer)
Products ( edit ) Zara stores have men 's and women 's clothing , as well as children 's clothing ( Zara Kids ) . Zara 's products are supplied based on
consumer trends . Its highly responsive supply chain ships new products to stores twice a week . After products are designed , they take ten to fifteen days to
reach the stores . All of the clothing is processed through the distribution center in Spain . New items are inspected , sorted , tagged , and loaded into
trucks . In most cases , the clothing is delivered within 48 hours . Zara produces over 450 million items per year . Manufacturing and distribution ( edit )
Zara Almere - Heren The men 's department of a typical Zara store . Almere , Netherlands Zara clothing made in Portugal Zara store in New York City . Zara store
in Brisbane Zara store in Palma de Mallorca ( Spain ) . Zara store in Brussels ( Belgium ) . Reportedly , Zara needs just one week to develop a new product and
get it to stores , compared to the six - month industry average , and launches around 12,000 new designs each year .
</item 2>

<item 3>
Zara (retailer)
Ortega opened additional stores throughout Spain . During the 1980s , Ortega changed the design , manufacturing , and distribution process to reduce lead times
and react to new trends in a quicker way , which he called `` instant fashions '' . The improvements included the use of information technologies and using
groups of designers instead of individuals . Zara 's previous logo Expansion ( edit ) In 1988 , the company started its international expansion through Porto ,
Portugal . In 1989 , it entered the United States , and then France in 1990 . During the 1990s , Zara expanded to Mexico ( 1992 ) , Greece , Belgium and Sweden
( 1993 ) . In the early 2000s , Zara opened its first stores in Japan and Singapore ( 2002 ) , Russia and Malaysia ( 2003 ) , China , Morocco , Estonia ,
Hungary and Romania ( 2004 ) , the Philippines , Costa Rica and Indonesia ( 2005 ) , South Korea ( 2008 ) , India ( 2010 ) , and South Africa and Australia (
2011 ) .
</item 3>

<item 4>
Zara (retailer)
as of 31 November 2017 : Africa South Africa : 9 Egypt : 5 Morocco : 4 Tunisia : 3 Algeria : 1 Americas United States : 85 Mexico : 84 Brazil : 56 Canada : 32
Colombia : 14 Argentina : 10 Chile : 9 Venezuela : 8 Peru : 4 Puerto Rico : 4 Dominican Republic : 3 Guatemala : 3 Panama : 3 Costa Rica : 2 Ecuador : 2 El
Salvador : 2 Honduras : 2 Uruguay : 2 Aruba : 1 Nicaragua : 1 Paraguay : 1 Asia China : 200 Japan : 99 South Korea : 43 Saudi Arabia : 39 Israel : 24 India : 20
Indonesia : 17 Hong Kong : 14 Thailand : 11 United Arab Emirates : 11 Malaysia : 10 Singapore : 9 Taiwan : 9 Philippines : 8 Lebanon : 7 Kazakhstan : 5 Kuwait :
5 Qatar : 5 Jordan : 3 Bahrain : 2 Macau : 2 Oman : 1 Vietnam : 1 Europe Spain : 430 France : 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United
</item 4>

<item 5>
Zara (retailer)
: 128 Italy : 104 Russia : 100 Portugal : 87 Germany : 77 United Kingdom : 64 Poland : 47 Greece : 46 Turkey : 41 Belgium : 31 Netherlands : 28 Romania : 24
Switzerland : 20 Austria : 14 Sweden : 12 Croatia : 10 Ireland : 9 Ukraine : 9 Hungary : 8 Bulgaria : 6 Cyprus : 6 Czech Republic : 6 Finland : 5 Lithuania : 5
Norway : 5 Serbia : 5 Slovenia : 5 Denmark : 4 Latvia : 4 Estonia : 3 Azerbaijan : 3 Bosnia and Herzegovina : 3 Georgia : 3 Luxembourg : 3 Slovakia : 3 Armenia
: 2 Albania : 1 Andorra : 1 Belarus : 1 Iceland : 1 Macedonia : 1 Malta : 1 Monaco : 1 Montenegro : 1 Oceania Australia : 19 New Zealand : 1 See also ( edit )
Spain portal Companies portal Fashion portal Bonds ( clothing ) Gap Inc . H&M List of companies of Spain European Retail Round Table References ( edit ) Jump up
^ `` Zara '' . inditex.com .
</item 5>

<item 6>
Zara (retailer)
Zara as of 2017 manages up to 20 clothing collections a year . Contents ( hide ) 1 History 1.1 Early history 1.2 Expansion 2 Products 3 Manufacturing and
distribution 3.1 Non-toxic clothing 4 Controversy 5 Stores 6 See also 7 References 8 External links History ( edit ) Early history ( edit ) Amancio Ortega
opened the first Zara store in 1975 in downtown A Coruña , Galicia , Spain . Ortega initially named the store Zorba after the classic film Zorba the Greek , but
after learning there was a bar with the same name two blocks away , they rearranged the letters molded for the sign to ' Zara . ' It is believed the extra ' a '
came from an additional set of letters that had been made for the company . The first store featured low - priced lookalike products of popular , higher - end
clothing fashions . Ortega opened additional stores throughout Spain .
</item 6>

<item 7>
Zara (retailer)
20 October 2014 . Graham Ruddick . accessed 5 April 2016 Jump up ^ Zara new season styles : what to buy at the Spanish giant right now . Linda Sharkey. 29 May
2015 . Independent . accessed 5 April 2016 Jump up ^ `` Presencia internacional '' . inditex.com. 30 April 2015 . Retrieved 9 September 2015 . Jump up ^ ``
Localizador de tiendas '' ( in Spanish ) . Zara.com . Retrieved 11 June 2015 . External links ( edit ) Official website Media related to Zara ( clothing ) at
Wikimedia Commons Retrieved from `` https://en.wikipedia.org/w/index.php?title=Zara_(retailer)&oldid=819154896 '' Categories : Clothing brands of Spain Clothing
retailers of Spain Companies based in Galicia Clothing companies established in 1974 Retail companies established in 1974 1974 establishments in Spain Inditex
brands Hidden categories : Pages using citations with accessdate and no URL CS1 Dutch - language sources ( nl ) CS1 Portuguese - language sources ( pt )
</item 7>

<item 8>
Zara (retailer)
On September 2010 , Zara launched its online boutique . The website began in Spain , the UK , Portugal , Italy , Germany and France . In November that same year
, Zara Online extended the service to five more countries : Austria , Ireland , the Netherlands , Belgium and Luxembourg . Online stores began operating in the
United States in 2011 , Russia and Canada in 2013 , and Mexico , Romania , and South Korea in 2014 . . India In 4 october 2017 . Zara introduced the use of RFID
technology in its stores in 2014 . The RFID chips are located in the security tags which are removed from clothing when it is purchased and can be reused . The
chip allows the company to quickly take inventory by detecting radio signals from the RFID tags . When an item is sold , the stockroom is immediately notified
so that the item can be replaced . An item that is not on the shelf can easily be found with the RFID tag . In 2015 , Zara was ranked 30 on Interbrand 's list
of best global brands .
</item 8>

<item 9>
Zara (retailer)
24 April 2005 . Retrieved 8 April 2016 . Jump up ^ `` Spains Zara Opens First Store in Korea '' . The Korea Times . 29 April 2008 . Retrieved 8 April 2016 .
Jump up ^ `` Fashion chain Zara opens its first Indian store '' . BBC . 31 May 2010 . Retrieved 8 April 2016 . Jump up ^ Emma Jordan ( 19 January 2011 ) . ``
Zara to Open in South Africa '' . iFashion . Retrieved 8 April 2016 . Jump up ^ `` Global phenomenon Zara finally opens in Sydney '' . Vogue. 19 April 2011 .
Retrieved 8 April 2016 . Jump up ^ `` Zara launches online boutique '' . fashionunited.com. 6 September 2010 . Retrieved 22 June 2014 . Jump up ^ `` Zara.com
launches in 5 more countries '' . Retrieved 30 May 2015 . Jump up ^ `` Inditex post record profits '' .
</item 9>

<item 10>
Zara (retailer)
When the news was eventually picked up by media outlets , and Inditex , Zara 's owning company , was asked to comment on the issue , Inditex replied that the
items in question have been suspended from sale , and that they are in contact with Bassen 's lawyer to clarify and address the issue . In November 2017 ,
customers shopping at Zara stores in Istanbul , Turkey found handwritten notes purportedly from Turkish workers in the pockets of in - store garments asking
shoppers to pressure Zara into paying them the wages they claim they are owed . Stores ( edit ) There are over 2,100 Zara stores located across 88 countries .
Zara usually selects the best located and most expensive real estate locations in the world to open its flagship stores . Zara 's flagship stores : Fifth Avenue
, New York Oxford Street , London Calle Serrano , Madrid Via del Corso , Rome Champs - Élysées , Paris Oscar Freire Street , Sao Paulo Corso Vittorio Emanuele ,
Milan Nevsky Prospect , Saint Petersburg Shibuya and Ginza , Tokyo Myeongdong in Seoul Flora Fountain in Mumbai The number of Zara stores in each country ,
</item 10>
=== END ITEM id=4cf251de90e9c8b4 ===
