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

This file has 5 items.

=== ITEM id=fbd37fa361de99d9 ===
Gold sentences and facts:
Sentence 0: Callability -- Some bonds give the issuer the right to repay the bond before the maturity date on the call dates ; see call option . These bonds are referred to as callable bonds . Most callable bonds allow the issuer to repay the bond at par . With some bonds , the issuer has to pay a premium , the so - called call premium . This is mainly the case for high - yield bonds . These have very strict covenants , restricting the issuer in its operations . To be free from these covenants , the issuer can repay the bonds early , but only at a high cost .
  Fact 0.0: Some bonds give the issuer the right to repay the bond before the maturity date on the call dates.
  Fact 0.1: Bonds that give the issuer this right are referred to as callable bonds.
  Fact 0.2: Most callable bonds allow the issuer to repay the bond at par.
  Fact 0.3: With some bonds, the issuer has to pay a premium, called the call premium.
  Fact 0.4: Call premiums are mainly the case for high-yield bonds.
  Fact 0.5: High-yield bonds have very strict covenants that restrict the issuer in its operations.
  Fact 0.6: To be free from these covenants, the issuer can repay high-yield bonds early.
  Fact 0.7: Repaying high-yield bonds early to be free from covenants is possible only at a high cost.

Retrieved items:
<item 1>
Bond (finance)
A European callable has only one call date . This is a special case of a Bermudan callable . An American callable can be called at any time until the maturity
date . A death put is an optional redemption feature on a debt instrument allowing the beneficiary of the estate of a deceased bondholder to put ( sell ) the
bond back to the issuer at face value in the event of the bondholder 's death or legal incapacitation . This is also known as a `` survivor 's option '' .
Sinking fund provision of the corporate bond indenture requires a certain portion of the issue to be retired periodically . The entire bond issue can be
liquidated by the maturity date ; if not , the remainder is called balloon maturity . Issuers may either pay to trustees , which in turn call randomly selected
bonds in the issue , or , alternatively , purchase bonds in open market , then return them to trustees . Bonds are often identified by its international
securities identification number , or ISIN , which is a 12 digit alphanumeric code that uniquely identifies debt securities .
</item 1>

<item 2>
Bond (finance)
Book - entry bond is a bond that does not have a paper certificate . As physically processing paper bonds and interest coupons became more expensive , issuers (
and banks that used to collect coupon interest for depositors ) have tried to discourage their use . Some book - entry bond issues do not offer the option of a
paper certificate , even to investors who prefer them . Lottery bonds are issued by European and other states . Interest is paid as on a traditional fixed rate
bond , but the issuer will redeem randomly selected individual bonds within the issue according to a schedule . Some of these redemptions will be for a higher
value than the face value of the bond . War bond is a bond issued by a government to fund military operations during wartime . This type of bond has low return
rate . Serial bond is a bond that matures in installments over a period of time . In effect , a $100,000 , 5 - year serial bond would mature in a $20,000
annuity over a 5 - year interval . Revenue bond is a special type of municipal bond distinguished by its guarantee of repayment solely from revenues generated
by a specified revenue - generating entity associated with the purpose of the bonds .
</item 2>

<item 3>
Bond (finance)
Optionality : Occasionally a bond may contain an embedded option ; that is , it grants option - like features to the holder or the issuer : Callability -- Some
bonds give the issuer the right to repay the bond before the maturity date on the call dates ; see call option . These bonds are referred to as callable bonds .
Most callable bonds allow the issuer to repay the bond at par . With some bonds , the issuer has to pay a premium , the so - called call premium . This is
mainly the case for high - yield bonds . These have very strict covenants , restricting the issuer in its operations . To be free from these covenants , the
issuer can repay the bonds early , but only at a high cost . Putability -- Some bonds give the holder the right to force the issuer to repay the bond before the
maturity date on the put dates ; see put option . These are referred to as retractable or putable bonds . Call dates and put dates -- the dates on which
callable and putable bonds can be redeemed early . There are four main categories : A Bermudan callable has several call dates , usually coinciding with coupon
dates .
</item 3>

<item 4>
Bond (finance)
A variation are stepped - coupon bonds , whose coupon increases during the life of the bond . Floating rate notes ( FRNs , floaters ) have a variable coupon
that is linked to a reference rate of interest , such as Libor or Euribor . For example , the coupon may be defined as three - month USD LIBOR + 0.20 % . The
coupon rate is recalculated periodically , typically every one or three months . Zero - coupon bonds ( zeros ) pay no regular interest . They are issued at a
substantial discount to par value , so that the interest is effectively rolled up to maturity ( and usually taxed as such ) . The bondholder receives the full
principal amount on the redemption date . An example of zero coupon bonds is Series E savings bonds issued by the U.S. government . Zero - coupon bonds may be
created from fixed rate bonds by a financial institution separating ( `` stripping off '' ) the coupons from the principal . In other words , the separated
coupons and the final principal payment of the bond may be traded separately . See IO ( Interest Only ) and PO ( Principal Only ) .
</item 4>

<item 5>
Bond (finance)
average life Yield curve Yield spread Yield to maturity Z - spread Securitized products Asset - backed security Collateralized debt obligation Collateralized
mortgage obligation Commercial mortgage - backed security Mortgage - backed security Bond options Callable bond Convertible bond Embedded option Exchangeable
bond Extendible bond Puttable bond Institutions Commercial Mortgage Securities Association ( CMSA ) International Capital Market Association ( ICMA ) Securities
Industry and Financial Markets Association ( SIFMA ) Corporate finance and investment banking Capital structure Convertible debt Exchangeable debt Mezzanine
debt Pari passu Preferred equity Second lien debt Senior debt Senior secured debt Shareholder loan Stock Subordinated debt Warrant Transactions ( terms /
conditions ) Equity offerings At - the - market offering Book building Bookrunner Corporate spin - off Equity carve - out Follow - on offering Greenshoe Reverse
Initial public offering Private placement Public offering Rights issue Seasoned equity offering Secondary market offering Underwriting Mergers and acquisitions
Buy side Control premium Demerger Divestment Drag - along right Management due diligence Managerial entrenchment Minority discount Pitch book Pre-emption right
Proxy fight Post-merger integration Sell side Shareholder rights plan Special - purpose entity Special situation Squeeze out Staggered board of directors Stock
swap Super-majority amendment Tag - along right
</item 5>

<item 6>
Bond (finance)
The most common types of bonds include municipal bonds and corporate bonds . The bond is a debt security , under which the issuer owes the holders a debt and (
depending on the terms of the bond ) is obliged to pay them interest ( the coupon ) or to repay the principal at a later date , termed the maturity date .
Interest is usually payable at fixed intervals ( semiannual , annual , sometimes monthly ) . Very often the bond is negotiable , that is , the ownership of the
instrument can be transferred in the secondary market . This means that once the transfer agents at the bank medallion stamp the bond , it is highly liquid on
the secondary market . Thus a bond is a form of loan or IOU : the holder of the bond is the lender ( creditor ) , the issuer of the bond is the borrower (
debtor ) , and the coupon is the interest . Bonds provide the borrower with external funds to finance long - term investments , or , in the case of government
bonds , to finance current expenditure . Certificates of deposit ( CDs ) or short - term commercial paper are considered to be money market instruments and not
bonds : the main difference is the length of the term of the instrument .
</item 6>

<item 7>
Bond (finance)
The net proceeds that the issuer receives are thus the issue price , less issuance fees . The market price of the bond will vary over its life : it may trade at
a premium ( above par , usually because market interest rates have fallen since issue ) , or at a discount ( price below par , if market rates have risen or
there is a high probability of default on the bond ) . Others ( edit ) Indentures and Covenants -- An indenture is a formal debt agreement that establishes the
terms of a bond issue , while covenants are the clauses of such an agreement . Covenants specify the rights of bondholders and the duties of issuers , such as
actions that the issuer is obligated to perform or is prohibited from performing . In the U.S. , federal and state securities and commercial laws apply to the
enforcement of these agreements , which are construed by courts as contracts between issuers and bondholders . The terms may be changed only with great
difficulty while the bonds are outstanding , with amendments to the governing document generally requiring approval by a majority ( or super-majority ) vote of
the bondholders .
</item 7>

<item 8>
Bond (finance)
The name `` coupon '' arose because in the past , paper bond certificates were issued which had coupons attached to them , one for each interest payment . On
the due dates the bondholder would hand in the coupon to a bank in exchange for the interest payment . Interest can be paid at different frequencies : generally
semi-annual , i.e. every 6 months , or annual . Bond issued by the Dutch East India Company in 1623 Yield ( edit ) The yield is the rate of return received from
investing in the bond . It usually refers either to The current yield , or running yield , which is simply the annual interest payment divided by the current
market price of the bond ( often the clean price ) . The yield to maturity , or redemption yield , which is a more useful measure of the return of the bond .
This takes into account the current market price , and the amount and timing of all remaining coupon payments and of the repayment due on maturity . It is
equivalent to the internal rate of return of a bond . Credit quality ( edit ) The quality of the issue refers to the probability that the bondholders will
receive the amounts promised at the due dates .
</item 8>

<item 9>
Bond (finance)
The latter are often issued in tranches . The senior tranches get paid back first , the subordinated tranches later . Covered bonds are backed by cash flows
from mortgages or public sector assets . Contrary to asset - backed securities the assets for such bonds remain on the issuers balance sheet . Perpetual bonds
are also often called perpetuities or ' Perps ' . They have no maturity date . The most famous of these are the UK Consols , which are also known as Treasury
Annuities or Undated Treasuries . Some of these were issued back in 1888 and still trade today , although the amounts are now insignificant . Some ultra-long -
term bonds ( sometimes a bond can last centuries : West Shore Railroad issued a bond which matures in 2361 ( i.e. 24th century ) are virtually perpetuities from
a financial point of view , with the current value of principal near zero . Bearer bond is an official certificate issued without a named holder . In other
words , the person who has the paper certificate can claim the value of the bond .
</item 9>

<item 10>
Bond (finance)
The length of time until the maturity date is often referred to as the term or tenure or maturity of a bond . The maturity can be any length of time , although
debt securities with a term of less than one year are generally designated money market instruments rather than bonds . Most bonds have a term of up to 30 years
. Some bonds have been issued with terms of 50 years or more , and historically there have been some issues with no maturity date ( irredeemable ) . In the
market for United States Treasury securities , there are three categories of bond maturities : short term ( bills ) : maturities between one and five years ;
medium term ( notes ) : maturities between six and twelve years ; long term ( bonds ) : maturities longer than twelve years . Coupon ( edit ) The coupon is the
interest rate that the issuer pays to the holder . Usually this rate is fixed throughout the life of the bond . It can also vary with a money market index ,
such as LIBOR , or it can be even more exotic .
</item 10>
=== END ITEM id=fbd37fa361de99d9 ===

=== ITEM id=8519fd0636f8d6b7 ===
Gold sentences and facts:
Sentence 0: Angular cheilitis can be caused by infection , irritation , or allergies . Infections include by fungi such as Candida albicans and bacteria such as Staph . aureus . Irritants include poorly fitting dentures , licking the lips or drooling , mouth breathing resulting in a dry mouth , sun exposure , overclosure of the mouth , smoking , and minor trauma . Allergies may include substances like toothpaste , makeup , and food . Often a number of factors are involved . Other factors may include poor nutrition or poor immune function . Diagnosis may be helped by testing for infections and patch testing for allergies .
  Fact 0.0: Angular cheilitis can be caused by infection.
  Fact 0.1: Angular cheilitis can be caused by irritation.
  Fact 0.2: Angular cheilitis can be caused by allergies.
  Fact 0.3: Infections that cause angular cheilitis include infection by fungi such as Candida albicans.
  Fact 0.4: Infections that cause angular cheilitis include infection by bacteria such as Staph. aureus.
  Fact 0.5: Irritants that cause angular cheilitis include poorly fitting dentures.
  Fact 0.6: Irritants that cause angular cheilitis include licking the lips or drooling.
  Fact 0.7: Irritants that cause angular cheilitis include mouth breathing resulting in a dry mouth.
  Fact 0.8: Irritants that cause angular cheilitis include sun exposure.
  Fact 0.9: Irritants that cause angular cheilitis include overclosure of the mouth.
  Fact 0.10: Irritants that cause angular cheilitis include smoking.
  Fact 0.11: Irritants that cause angular cheilitis include minor trauma.
  Fact 0.12: Allergies that cause angular cheilitis may include substances like toothpaste.
  Fact 0.13: Allergies that cause angular cheilitis may include substances like makeup.
  Fact 0.14: Allergies that cause angular cheilitis may include substances like food.
  Fact 0.15: Often a number of factors are involved in angular cheilitis.
  Fact 0.16: Other factors may include poor nutrition.
  Fact 0.17: Other factors may include poor immune function.
  Fact 0.18: Diagnosis of angular cheilitis may be helped by testing for infections.
  Fact 0.19: Diagnosis of angular cheilitis may be helped by patch testing for allergies.

Retrieved items:
<item 1>
Angular cheilitis
It is thought that reduced vertical dimension of the lower face may be a contributing factor in up to 11 % of elderly persons with angular cheilitis and in up
to 18 % of denture wearers who have angular cheilitis . Reduced vertical dimension can also be caused by tooth migration , wearing orthodontic appliances , and
elastic tissue damage caused by ultraviolet light exposure and smoking . Habits or conditions that keep the corners of the mouth moist might include chronic lip
licking , thumb sucking ( or sucking on other objects such as pens , pipes , lollipops ) , dental cleaning ( e.g. flossing ) , chewing gum , hypersalivation ,
drooling and mouth breathing . Some consider habitual lip licking or picking to be a form of nervous tic , and do not consider this to be true angular cheilitis
, instead calling it perlèche ( derived from the French word pourlècher meaning `` to lick one 's lips '' ) , or `` factitious cheilitis '' is applied to this
habit .
</item 1>

<item 2>
Angular cheilitis
In chronic angular cheilitis , there may be suppuration ( pus formation ) , exfoliation ( scaling ) and formation of granulation tissue . Sometimes contributing
factors can be readily seen , such as loss of lower face height from poorly made or worn dentures , which results in mandibular overclosure ( `` collapse of
jaws '' ) . If there is a nutritional deficiency underlying the condition , various other signs and symptoms such as glossitis ( swollen tongue ) may be present
. In people with angular cheilitis who wear dentures , often there may be erythematous mucosa underneath the denture ( normally the upper denture ) , an
appearance consistent with denture - related stomatitis . Typically the lesions give symptoms of soreness , pain , pruritus ( itching ) or burning or a raw
feeling . Causes ( edit ) Angular cheilitis is thought to be multifactorial disorder of infectious origin , with many local and systemic predisposing factors .
The sores in angular cheilitis are often infected with fungi ( yeasts ) , bacteria , or a combination thereof ; this may represent a secondary , opportunistic
infection by these pathogens .
</item 2>

<item 3>
Angular cheilitis
Infections include by fungi such as Candida albicans and bacteria such as Staph . aureus . Irritants include poorly fitting dentures , licking the lips or
drooling , mouth breathing resulting in a dry mouth , sun exposure , overclosure of the mouth , smoking , and minor trauma . Allergies may include substances
like toothpaste , makeup , and food . Often a number of factors are involved . Other factors may include poor nutrition or poor immune function . Diagnosis may
be helped by testing for infections and patch testing for allergies . Treatment for angular cheilitis is typically based on the underlying causes along with the
use of a barrier cream . Frequently an antifungal and antibacterial cream is also tried . Angular cheilitis is a fairly common problem , with estimates that it
affects 0.7 % of the population . It occurs most often in the 30s to 60s , although is also relatively common in children . In the developing world , iron and
vitamin deficiencies are a common cause .
</item 3>

<item 4>
Angular cheilitis
In the developing world , iron and vitamin deficiencies are a common cause . Contents ( hide ) 1 Signs and symptoms 2 Causes 2.1 Infection 2.2 Irritation
contact dermatitis 2.3 Nutritional deficiencies 2.4 Systemic disorders 2.5 Drugs 2.6 Allergic contact dermatitis 3 Diagnosis 3.1 Classification 4 Management 5
Prognosis 6 Epidemiology 7 References 8 External links Signs and symptoms ( edit ) Angular cheilitis -- a fissure running in the corner of the mouth with
reddened , irritated facial skin adjacent . A fairly mild case of angular cheilitis extending onto the facial skin in a young person ( affected area is within
the black oval ) . Angular cheilitis is a fairly non specific term which describes the presence of an inflammatory lesion in a particular anatomic site ( i.e.
the corner of the mouth ) . As there are different possible causes and contributing factors from one person to the next , the appearance of the lesion is
somewhat variable . The lesions are more commonly symmetrically present on both sides of the mouth , but sometimes only one side may be affected .
</item 4>

<item 5>
Angular cheilitis
A cold sore at the corner of the mouth behaves similarly to elsewhere on the lips , and follows a pattern of vesicle ( blister ) formation followed by rupture
leaving a crusted sore which resolves in about 7 -- 10 days , and recurs in the same spot periodically , especially during periods of stress . Rather than
utilizing antifungal creams , angular herpes simplex is treated in the same way as a cold sore , with topical antiviral drugs such as aciclovir . Irritation
contact dermatitis ( edit ) A famous sketch by Leonardo da Vinci in preparation to depict the face of Judas Iscariot in The Last supper . The subject shows
overclosure of the jaws and loss of facial support around the mouth . Pronounced skin folds extending from the corner of the mouth . Main article : Irritant
contact dermatitis 22 % of cases of angular cheilitis are due to irritants . Saliva contains digestive enzymes , which may have a degree of digestive action on
tissues if they are left in contact . The corner of the mouth is normally exposed to saliva more than any other part of the lips .
</item 5>

<item 6>
Angular cheilitis
The corner of the mouth is normally exposed to saliva more than any other part of the lips . Reduced lower facial height ( vertical dimension or facial support
) is usually caused by edentulism ( tooth loss ) , or wearing worn down , old dentures or ones which are not designed optimally . This results in overclosure of
the mandible ( collapse of the jaws ) , which extenuates the angular skin folds at the corners of the mouth , in effect creating an intertriginous skin crease .
The tendency of saliva to pool in these areas is increased , constantly wetting the area , which may cause tissue maceration and favors the development of a
yeast infection . As such , angular cheilitis is more commonly seen in edentulous people ( people without any teeth ) . It is by contrast uncommon in persons
who retain their natural teeth . Angular cheilitis is also commonly seen in denture wearers . Angular cheilitis is present in about 30 % of people with denture
- related stomatitis .
</item 6>

<item 7>
Angular cheilitis
Examples include cocaine , methamphetamines , heroin , and hallucinogens . Allergic contact dermatitis ( edit ) See also : Allergic contact cheilitis , Allergic
contact dermatitis , and Allergic contact stomatitis Patch test Allergic reactions may account for about 25 -- 34 % of cases of generalized cheilitis ( i.e. ,
inflammation not confined to the angles of the mouth ) . It is unknown how frequently allergic reactions are responsible for cases of angular cheilitis , but
any substance capable of causing generalized allergic cheilitis may present involving the corners of the mouth alone . Examples of potential allergens include
substances that may be present in some types of lipstick , toothpaste , acne products , cosmetics , chewing gum , mouthwash , foods , dental appliances , and
materials from dentures or mercury containing amalgam fillings . It is usually impossible to tell the difference between irritant contact dermatitis and
allergic contact dermatitis without a patch test . Diagnosis ( edit ) Photographic Comparison of : 1 ) a Canker Sore - inside the mouth , 2 ) Herpes , 3 )
Angular Cheilitis and 4 ) Chapped Lips .
</item 7>

<item 8>
Angular cheilitis
The term `` cheilocandidiasis '' describes exfoliative ( flaking ) lesions of the lips and the skin around the lips , and is caused by a superficial candidal
infection due to chronic lip licking . Less severe cases occur during cold , dry weather , and is a form of chapped lips . Individuals may lick their lips in an
attempt to provide a temporary moment of relief , only serving to worsen the condition . The sunscreen in some types of lip balm degrades over time into an
irritant . Using expired lipbalm can initiate mild angular cheilitis , and when the person applies more lipbalm to alleviate the cracking , it only aggravates
it . Because of the delayed onset of contact dermatitis and the recovery period lasting days to weeks , people typically do not make the connection between the
causative agent and the symptoms . Nutritional deficiencies ( edit ) Several different nutritional deficiency states of vitamins or minerals have been linked to
AC . It is thought that in about 25 % of people with AC , iron deficiency or deficiency of B vitamins are involved . Nutritional deficiencies may be a more
common cause of AC in Third World countries .
</item 8>

<item 9>
Angular cheilitis
Such examples include people with anorexia nervosa . Other disorders may cause lip enlargement ( e.g. orofacial granulomatosis ) , which alters the local
anatomy and extenuates the skin folds at the corners of the mouth . More still may be involved because they affect the immune system , allowing normally
harmless organisms like Candida to become pathogenic and cause an infection . Xerostomia ( dry mouth ) is thought to account for about 5 % of cases of AC .
Xerostomia itself has many possible causes , but commonly the cause may be side effects of medications , or conditions such as Sjögren 's syndrome . Conversely
, conditions which cause drooling or sialorrhoea ( excessive salivation ) can cause angular cheilitis by creating a constant wet environment in the corners of
the mouth . About 25 % of people with Down syndrome appear to have AC . This is due to relative macroglossia , an apparently large tongue in a small mouth ,
which may constantly stick out of the mouth causing maceration of the corners of the mouth with saliva .
</item 9>

<item 10>
Angular cheilitis
Angular cheilitis - wikipedia Angular cheilitis Jump to : navigation , search Angular cheilitis Synonyms Rhagades , perlèche , cheilosis , angular cheilosis ,
commissural cheilitis , angular stomatitis Bilateral angular cheilitis in an elderly individual with false teeth , iron deficiency anemia and dry mouth
Pronunciation / kaɪˈlaɪtɪs / Specialty Dermatology Symptoms Redness , skin breakdown and crusting at the corner of the mouth Usual onset Children , 30s to 60s
Duration Days to years Causes Infection , irritation , allergies Treatment Based on cause , barrier cream Frequency 0.7 % of the population Angular cheilitis (
AC ) , is inflammation of one or both corners of the mouth . Often the corners are red with skin breakdown and crusting . It can also be itchy or painful . The
condition can last for days to years . Angular cheilitis is a type of cheilitis ( inflammation of the lips ) . Angular cheilitis can be caused by infection ,
irritation , or allergies . Infections include by fungi such as Candida albicans and bacteria such as Staph .
</item 10>
=== END ITEM id=8519fd0636f8d6b7 ===

=== ITEM id=15990c42e8db0204 ===
Gold sentences and facts:
Sentence 0: Urdu lyrics كشورِ حسين شاد باد ‬ پاک سرزمین شاد باد ‬ ارضِ پاکستان ! ‏ ‬ تُو نشانِ عزمِ عالی شان ‬ مرکزِ یقین شاد باد ‬ قوّتِ اُخوّتِ عوام ‬ پاک سرزمین کا نظام ‬ پائنده تابنده باد ! ‏ ‬ قوم ، ملک ، سلطنت ‬ شاد باد منزلِ مراد ‬ رہبرِ ترقّی و کمال ‬ پرچمِ ستاره و ہلال ‬ جانِ استقبال ! ‏ ‬ ترجمانِ ماضی ، شانِ حال ‬ سایۂ خدائے ذوالجلال ‬ Transliteration ( ALA - LC ) Translation ( poetical ) Translation ( literal ) Pāk sarzamīn shād bād Kishwar - i ḥasīn shād bād Tū nishān - i ʿazm - i ʿālī shān Arẓ - i Pākistān ! Markaz - i yaqīn shād bād May the holy land , stay glad ; Beautious realm , stay glad . Thou , the sign of high resolve -- O Land of Pakistan ! Citadel of faith , stay glad . Blessed be the sacred land , Happy be the bounteous realm . Thou symbol of high resolve , O Land of Pakistan ! Blessed be the citadel of faith . Pāk sarzamīn kā niz̤ām Quwwat - i Ukhuwwat - i ʿawām Qaum , mulk , salt̤anat Pāyindah tābindah bād ! Shād bād manzil - i murād Order of the holy land , Power of fraternity of the populace ; The nation , country , and domain ; Ever luminous remain ! The cherished goal , stay glad . The order of this sacred land , The might of the brotherhood of the people , May the nation , the country , and the state , Shine in glory everlasting ! Blessed be the goal of our ambition . Parcam - i sitārah o - hilāl Rahbar - i taraqqī o - kamāl Tarjumān - i māẓī , shān - i ḥāl Jān - i istiqbāl ! Sāyah - yi Khudā - yi Ẕū l - jalāl Flag with the star and crescent , The leader of progress and ascent , Dragoman of past , the pride of present ; Soul of the future ! Shadow of the God of grandeur . The flag of the crescent and star , Leads the way to progress and perfection , Interpreter of our past , glory of our present , inspiration for our future ! Shade of God , the Glorious and Mighty .
  Fact 0.0: The text gives Urdu lyrics about the Land of Pakistan.
  Fact 0.1: The text gives an ALA-LC transliteration of the Urdu lyrics.
  Fact 0.2: The text gives a poetical English translation of the lyrics.
  Fact 0.3: The text gives a literal English translation of the lyrics.
  Fact 0.4: The lyrics wish that the holy land stay glad.
  Fact 0.5: The lyrics wish that the beautiful realm stay glad.
  Fact 0.6: The lyrics describe the Land of Pakistan as the sign of high resolve.
  Fact 0.7: The lyrics describe the Land of Pakistan as the citadel of faith.
  Fact 0.8: The lyrics refer to the order of the holy land.
  Fact 0.9: The lyrics refer to the power of the brotherhood of the people.
  Fact 0.10: The lyrics wish that the nation, the country, and the state shine in everlasting glory.
  Fact 0.11: The lyrics wish that the cherished goal stay glad.
  Fact 0.12: The lyrics refer to the flag with the star and crescent.
  Fact 0.13: The lyrics describe the flag as the leader of progress and ascent.
  Fact 0.14: The lyrics describe the flag as an interpreter of the past.
  Fact 0.15: The lyrics describe the flag as the pride of the present.
  Fact 0.16: The lyrics describe the flag as the soul of the future.
  Fact 0.17: The lyrics describe the land as the shade of God, the Glorious and Mighty.

Retrieved items:
<item 1>
Qaumi Taranah
Music ( edit ) The Qaumi Taranah is a melodious and harmonious rendering of a three - stanza composition with a tune based on eastern music but arranged in such
a manner that it can be easily played by foreign bands . The music , composed by the Pakistani musician and composer , Ahmad G. Chagla in 1949 , reflects his
background in both eastern and western music . Twenty - one musical instruments and thirty - eight different tones are used to play the Qaumi Taranah , the
duration of which is 80 seconds . Lyrics ( edit ) The Urdu lyrics , written by the Pakistani Urdu - language poet , Hafeez Jullundhri in 1952 , have commonality
with Persian , making them understandable in both languages . No verse in the three stanza lyrics is repeated . The anthem has heavy Persian poetic vocabulary
and only uses one exclusively Urdu word ' kā ' . Urdu lyrics كشورِ حسين شاد باد ‬ پاک سرزمین شاد باد ‬ ارضِ پاکستان !
</item 1>

<item 2>
Qaumi Taranah
rɑː. nɑ ) , lit . `` National Anthem '' ) , also known as Pāk Sarzamīn ( Urdu : پاک سرزمین ‬ ‎ , pronounced ( ˈpɑːk ˈsər. zə. miːn ) , lit . `` The Sacred Land
'' ) , is the national anthem of Pakistan . Its music was composed by Ahmad G. Chagla in 1949 , preceding the lyrics , which were written by Hafeez Jullundhri
in 1952 . It was officially adopted as Pakistan 's national anthem in August 1954 and was recorded in the same year by eleven major singers of Pakistan
including Ahmad Rushdi , Kaukab Jahan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zawar Hussain , Akhtar Abbas , Ghulam Dastagir , Anwar Zaheer and Akhtar
Wasi Ali .
</item 2>

<item 3>
Qaumi Taranah
Retrieved 1 September 2016 . Jump up ^ National Anthem. http://anisshakur.tripod.com/id129.html Jump up ^ Michael Jamieson Bristow , National-Anthems.org . ``
Forty National Anthems '' . Retrieved 2006 - 04 - 12 . ^ Jump up to : Aqeel Abbas Jafari ( 2010 ) . Pakistan Chronicle ( in Urdu ) ( 1st ed . ) . 94 / 1 , 26th
St. , Ph. 6 , D.H.A. , Karachi , Pakistan : Wirsa Publishers . p. 42 . ISBN 9789699454004 . ^ Jump up to : `` National Anthem '' . Ministry of Information and
Broadcasting , Government of Pakistan . Archived from the original on 2007 - 10 - 26 . Retrieved 2007 - 11 - 29 . Jump up ^ Mazhar Iqbal , Mazhar.dk . `` Ahmad
Rushdi '' . http://mazhar.dk/film/singers/ahmadrushdi/ .
</item 3>

<item 4>
Qaumi Taranah
Official recognition to the national anthem , however , was not given until August 1954 . The NAC distributed records of the composed tune amongst prominent
poets , who responded by writing and submitting several hundred songs for evaluation by the NAC . Eventually , the lyrics written by Hafeez Jullundhri were
approved and the new national anthem was broadcast publicly for the first time on Radio Pakistan on 13 August 1954 , sung by Hafeez Jullundhri himself .
Official approval was announced by the Ministry of Information and Broadcasting on 16 August 1954 . The composer , Ahmed G. Chagla , died in 1953 , before the
new national anthem was officially adopted . In 1955 , there was a performance of the national anthem involving 11 major singers of Pakistan , including Ahmad
Rushdi , Kaukab Jahan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zawar Hussain , Akhtar Abbas , Ghulam Dastagir , Anwar Zaheer and Akhtar Wasi Ali .
</item 4>

<item 5>
Qaumi Taranah
1954 -- Officially adopted as the national anthem and broadcast for the first time on Radio Pakistan on 13 August 1955 -- Sung by 11 famous Pakistani singers
including Ahmad Rushdi , Shamim Bano , Kokab Jehan , Rasheeda Begum , Najam Ara , Naseema Shaheen , Zwar Hussain , Akhtar Abbas , Ghulam Dastgir , Anwar Zaheer
and Akhtar Wassi 1996 -- Rendered in electric guitar for the first time by Pakistani rock band Junoon in their album Inquilaab 2009 -- Rendered as an acoustic
instrumental for the first time by Pakistani musician Jehangir Aziz Hayat 2011 -- On 14 August , 5,857 people gathered in a stadium in Karachi to sing the Qaumi
Taranah and set a new world record for most people gathered to sing a national anthem simultaneously . 2012 -- On 20 October , 42,813 people gathered in a
stadium in Lahore to sing the Qaumi Taranah and set a new world record for most people gathered to sing a national anthem simultaneously , which was certified
by Guinness World Records .
</item 5>

<item 6>
Qaumi Taranah
Archived from the original on 2007 - 10 - 26 . Retrieved 2007 - 12 - 09 . National Anthem of Pakistan Pakistan National Songs National symbols of Pakistan Main
symbols Qaumi Taranah ( national anthem ) State emblem of Pakistan Flag of Pakistan Iman , Ittihad , Nazm ( motto ) Lahore Resolution ( national document ) Urdu
( national language ) People Muhammad Ali Jinnah ( Father of the Nation ) Muhammad Iqbal ( National poet ) Fatima Jinnah ( Mother of the Nation ) Other symbols
Cedrus deodara ( national tree ) Chukar partridge ( national bird ) Jasminum officinale ( national flower ) Markhor ( national animal ) Structures Bab -
e-Khyber ( monument ) Faisal Mosque Mazar - e-Quaid ( mausoleum ) Minar - e-Pakistan ( monument ) Pakistan Monument ( monument ) Quaid - e-Azam Residency
National anthems of Asia National Abkhazia Afghanistan Armenia Artsakh ( Nagorno - Karabakh ) Azerbaijan Bahrain Bangladesh Bhutan Brunei Cambodia China Cyprus
East Timor Egypt Georgia India Indonesia Iran Iraq Israel Japan Jordan Kazakhstan North Korea South Korea
</item 6>

<item 7>
Qaumi Taranah
Qaumi Taranah - wikipedia Qaumi Taranah Jump to : navigation , search Qaumi Taranah قومی ترانہ English : National Anthem of Pakistan National Anthem National
anthem of Pakistan Lyrics Hafeez Jullundhri , June 1952 Music Ahmad G. Chagla , 21 August 1949 Adopted 13 August 1954 Relinquished Present Audio sample Qaumi
Taranah ( Instrumental ) file help Ahmad G. Chagla composed the music of the National Anthem of Pakistan in 1949 Ahmed Rushdi recorded the National Anthem of
Pakistan in 1954 The Qaumi Taranah ( Urdu : قومی ترانہ ‬ ‎ , Qaumī Tarānah pronounced ( ˈqɔː. mi ˈt̪ə. rɑː. nɑ ) , lit .
</item 7>

<item 8>
Qaumi Taranah
http://mazhar.dk/film/singers/ahmadrushdi/ . Retrieved 2006 - 04 - 12 . Jump up ^ Mazhar Iqbal , Mazhar.dk . `` National Anthem of Pakistan '' . Retrieved 2006
- 04 - 12 . Jump up ^ Information Ministry , Government of Pakistan . `` Basic Facts '' . Retrieved 2006 - 04 - 12 . Jump up ^ ( 1 ) , National Anthem of
Pakistan Jump up ^ ( 2 ) , The National Anthem of Pakistan Jump up ^ ( 3 ) , Switch to Urdu from English as official language a hurdle for Pakistan Jump up ^ (
4 ) , History can not be falsely written and manufactured Jump up ^ ( 5 ) , Pakistan Basic Facts Jump up ^ Pasha , Muhammad A. English Composition ( Part II ) .
Lahore : Command Publications . Jump up ^ `` Pakistan creates new anthem record '' . The Express Tribune . 2011 - 08 - 12 . Retrieved 2012 - 01 - 01 . Jump up ^
`` Pakistan reclaims anthem singing record '' .
</item 8>

<item 9>
Qaumi Taranah
Contents ( hide ) 1 History 2 Music 3 Lyrics 4 Milestones 5 See also 6 References 7 External links History ( edit ) In early 1948 , A.R. Ghani , a Muslim from
Transvaal , South Africa , offered two prizes of five thousand rupees each for the poet and composer of a new national anthem for the newly independent state of
Pakistan . The prizes were announced through a government press advertisement published in June 1948 . In December 1948 , the Government of Pakistan established
the National Anthem Committee ( NAC ) with the task of coming up with the composition and lyrics for the official national anthem of Pakistan . The NAC was
initially chaired by the Information Secretary , Sheikh Muhammad Ikram , and its members included several politicians , poets and musicians , including Abdur
Rab Nishtar , Ahmad G. Chagla and Hafeez Jullundhri . The NAC encountered early difficulties in finding suitable music and lyrics . When President Sukarno of
Indonesia became the first foreign head of state to visit Pakistan on 30 January 1950 , there was no Pakistani national anthem to be played .
</item 9>

<item 10>
Qaumi Taranah
BBC News . 2017 - 08 - 14 . Retrieved 2017 - 08 - 15 . Jump up ^ `` Indian A capella group sings Pakistan 's national anthem on 71st Independence Day '' .
Firstpost. 2017 - 08 - 14 . Retrieved 2017 - 08 - 15 . Jump up ^ Staff , Images ( 2017 - 08 - 12 ) . `` Indians sing the Pakistani national anthem as a gift for
our Independence Day '' . Images . Retrieved 2017 - 08 - 15 . External links ( edit ) This article contains Urdu text . Without proper rendering support , you
may see unjoined letters running left to right or other symbols instead of Urdu script . Wikimedia Commons has media related to : Qaumi Taranah ( category ) Pak
Sar Zameen Shad Bad Lyrics Pakistan : Qaumī Tarānah -- Audio of the national anthem of Pakistan , with information and lyrics Ministry of Information and
Broadcasting , Government of Pakistan . `` National Anthem '' . Archived from the original on 2007 - 10 - 26 .
</item 10>
=== END ITEM id=15990c42e8db0204 ===

=== ITEM id=79cc17e5367ed7db ===
Gold sentences and facts:
Sentence 0: Arthur `` The President '' Flanders -- An inmate on death row , convicted of killing his father in an insurance - fraud scheme . His sentence is commuted to life imprisonment , but he is later murdered by an unknown inmate in the prison laundry . Arlen `` The Chief '' Bitterbuck -- A Washita Cherokee death - row inmate , convicted of killing a man in a drunken brawl over a pair of boots . His execution is the first of three mentioned in Paul 's story . Janice Edgecombe -- Paul Edgecombe 's wife . Dies in a bus accident on the way to her grandchild 's graduation in 1956 . Hal Moores -- The warden at Cold Mountain Penitentiary . Melinda Moores -- Warden Moores 's wife , who is dying of a brain tumor and is cured by John Coffey . She eventually dies of a heart attack ten years later . Curtis Anderson -- The assistant warden . Dean Stanton -- A guard on E Block who is strangled and nearly killed by William Wharton . A father of young children , he takes no part in the taking of John Coffey to Melinda Moores due to the risk of losing his job . He applies for relocation to C Block after John Coffey 's death , where he is murdered by an inmate 4 months later . Harry Terwilliger -- One of the main guards on E Block along with his friends Paul Edgecombe , Brutus Howell and Dean Stanton . He eventually dies of cancer , long after John Coffey 's execution . Bill Dodge -- A `` floater '' guard on E Block ( not permanently assigned there ) . Jack Van Hay -- A guard who is part of the execution team . He operates the switch room . Toot - Toot -- A trustee who stands in for the condemned during execution rehearsals and sells snacks to prisoners and guards . Burt Hammersmith -- A reporter who wrote on the Detterick twins ' murders and John Coffey 's trial . Despite believing himself to be an `` Enlightenment '' man , he displays prejudice in his stance on `` Negroes '' and tries to convince Paul of John Coffey 's guilt . Elaine Connelly -- A friend of Paul in the present - day nursing home where he tells his story . She is later revealed to be the grandmother of the Speaker of the Georgia House of Representatives , using this position to scare Brad Dolan from harassing Paul . Homer Cribus -- The Sheriff of Trapingus County , where the murders of the Detterick twins took place . Despite playing no part in apprehending John Coffey , he later shows up at the execution . An outspoken Baptist with strong racial prejudice and immensely overweight , he later succumbs to a heart attack while having sex with a 17 - year - old African American in his office . Rob McGee -- The Deputy Sheriff of Trapingus County , who led the search party which found John Coffey . While displaying strong doubt about Coffey 's guilt after being shown signs of innocence by Paul Edgecombe , he is still powerless to call for an appeal as he is subordinate to Sheriff Cribus , whom he allegedly hopes to succeed . Brad Dolan -- A malicious nursing home employee who harasses Paul Edgecombe . Paul strongly compares him to Percy Wetmore , and several times mistakes him for Percy , despite the latter dying in 1965 . Kathe and Cora Detterick -- The two young girls whom John Coffey was convicted of raping and murdering . Killed by William Wharton . Klaus and Marjorie Detterick -- The parents of Kathe and Cora .
  Fact 0.0: Arthur 'The President' Flanders is an inmate on death row.
  Fact 0.1: Arthur Flanders was convicted of killing his father in an insurance-fraud scheme.
  Fact 0.2: Arthur Flanders's sentence was commuted to life imprisonment.
  Fact 0.3: Arthur Flanders was later murdered by an unknown inmate.
  Fact 0.4: Arthur Flanders was murdered in the prison laundry.
  Fact 0.5: Arlen 'The Chief' Bitterbuck is a Washita Cherokee death-row inmate.
  Fact 0.6: Arlen Bitterbuck was convicted of killing a man in a drunken brawl over a pair of boots.
  Fact 0.7: Arlen Bitterbuck's execution is the first of three mentioned in Paul's story.
  Fact 0.8: Janice Edgecombe is Paul Edgecombe's wife.
  Fact 0.9: Janice Edgecombe dies in a bus accident on the way to her grandchild's graduation.
  Fact 0.10: Janice Edgecombe's death occurred in 1956.
  Fact 0.11: Hal Moores is the warden at Cold Mountain Penitentiary.
  Fact 0.12: Melinda Moores is Warden Moores's wife.
  Fact 0.13: Melinda Moores is dying of a brain tumor.
  Fact 0.14: John Coffey cures Melinda Moores of the brain tumor.
  Fact 0.15: Melinda Moores dies of a heart attack ten years later.
  Fact 0.16: Curtis Anderson is the assistant warden.
  Fact 0.17: Dean Stanton is a guard on E Block.
  Fact 0.18: William Wharton strangles Dean Stanton and nearly kills him.
  Fact 0.19: Dean Stanton is a father of young children.
  Fact 0.20: Dean Stanton takes no part in taking John Coffey to Melinda Moores due to the risk of losing his job.
  Fact 0.21: Dean Stanton applies for relocation to C Block after John Coffey's death.
  Fact 0.22: Dean Stanton is murdered by an inmate on C Block 4 months later.
  Fact 0.23: Harry Terwilliger is one of the main guards on E Block.
  Fact 0.24: Harry Terwilliger is friends with Paul Edgecombe, Brutus Howell, and Dean Stanton.
  Fact 0.25: Harry Terwilliger dies of cancer.
  Fact 0.26: Harry Terwilliger's death came long after John Coffey's execution.
  Fact 0.27: Bill Dodge is a 'floater' guard on E Block.
  Fact 0.28: Bill Dodge is not permanently assigned to E Block.
  Fact 0.29: Jack Van Hay is a guard who is part of the execution team.
  Fact 0.30: Jack Van Hay operates the switch room.
  Fact 0.31: Toot-Toot is a trustee.
  Fact 0.32: Toot-Toot stands in for the condemned during execution rehearsals.
  Fact 0.33: Toot-Toot sells snacks to prisoners and guards.
  Fact 0.34: Burt Hammersmith is a reporter.
  Fact 0.35: Burt Hammersmith wrote on the Detterick twins' murders.
  Fact 0.36: Burt Hammersmith wrote on John Coffey's trial.
  Fact 0.37: Burt Hammersmith believes himself to be an 'Enlightenment' man.
  Fact 0.38: Burt Hammersmith displays prejudice in his stance on 'Negroes'.
  Fact 0.39: Burt Hammersmith tries to convince Paul Edgecombe of John Coffey's guilt.
  Fact 0.40: Elaine Connelly is a friend of Paul Edgecombe in the present-day nursing home where he tells his story.
  Fact 0.41: Elaine Connelly is later revealed to be the grandmother of the Speaker of the Georgia House of Representatives.
  Fact 0.42: Elaine Connelly uses this position to scare Brad Dolan from harassing Paul.
  Fact 0.43: Homer Cribus is the Sheriff of Trapingus County.
  Fact 0.44: The Detterick twins' murders took place in Trapingus County.
  Fact 0.45: Homer Cribus played no part in apprehending John Coffey.
  Fact 0.46: Homer Cribus later shows up at the execution.
  Fact 0.47: Homer Cribus is an outspoken Baptist.
  Fact 0.48: Homer Cribus has strong racial prejudice.
  Fact 0.49: Homer Cribus is immensely overweight.
  Fact 0.50: Homer Cribus later succumbs to a heart attack while having sex with a 17-year-old African American in his office.
  Fact 0.51: Rob McGee is the Deputy Sheriff of Trapingus County.
  Fact 0.52: Rob McGee led the search party which found John Coffey.
  Fact 0.53: Rob McGee displays strong doubt about Coffey's guilt after being shown signs of innocence by Paul Edgecombe.
  Fact 0.54: Rob McGee is powerless to call for an appeal.
  Fact 0.55: Rob McGee is subordinate to Sheriff Cribus.
  Fact 0.56: Rob McGee allegedly hopes to succeed Sheriff Cribus.
  Fact 0.57: Brad Dolan is a malicious nursing home employee.
  Fact 0.58: Brad Dolan harasses Paul Edgecombe.
  Fact 0.59: Paul Edgecombe strongly compares Brad Dolan to Percy Wetmore.
  Fact 0.60: Paul Edgecombe several times mistakes Brad Dolan for Percy Wetmore.
  Fact 0.61: Percy Wetmore died in 1965.
  Fact 0.62: Kathe Detterick and Cora Detterick are two young girls.
  Fact 0.63: John Coffey was convicted of raping and murdering Kathe and Cora Detterick.
  Fact 0.64: William Wharton killed Kathe and Cora Detterick.
  Fact 0.65: Klaus Detterick and Marjorie Detterick are the parents of Kathe and Cora Detterick.
Sentence 1: Toot - Toot -- A trustee who stands in for the condemned during execution rehearsals and sells snacks to prisoners and guards .
  Fact 1.0: Toot-Toot is a trustee.
  Fact 1.1: Toot-Toot stands in for the condemned during execution rehearsals.
  Fact 1.2: Toot-Toot sells snacks to prisoners and guards.

Retrieved items:
<item 1>
The Green Mile (novel)
He is punished by being placed in solitary confinement , but never seems to learn his lesson . As John Coffey is being smuggled to Hal Moores 's house , Wild
Bill grabs his arm and Coffey sees that he actually committed the murders Coffey was accused of . Therefore , Coffey gives Percy the `` sickness '' he took from
the warden 's wife , causing him to kill Wharton . Eduard `` Del '' Delacroix -- He is a Cajun prisoner with a fairly slow grasp of the English language . He is
incarcerated for multiple deaths due to a fire he started while trying to cover up his rape and murder of a young girl . While on the Mile , Del befriends a
mouse named Mr. Jingles , who becomes his best friend in his last days on death row . Percy , his enemy , sabotages his execution , causing Del to die in a slow
, gruesome death in the electric chair . Mr. Jingles ( nicknamed `` Steamboat Willie '' ) -- An unusually intelligent mouse who enjoys eating peppermint sweets
. He becomes a friend to Eduard Delacroix in the few days before the man is executed .
</item 1>

<item 2>
The Green Mile (novel)
Curtis Anderson -- The assistant warden . Dean Stanton -- A guard on E Block who is strangled and nearly killed by William Wharton . A father of young children
, he takes no part in the taking of John Coffey to Melinda Moores due to the risk of losing his job . He applies for relocation to C Block after John Coffey 's
death , where he is murdered by an inmate 4 months later . Harry Terwilliger -- One of the main guards on E Block along with his friends Paul Edgecombe , Brutus
Howell and Dean Stanton . He eventually dies of cancer , long after John Coffey 's execution . Bill Dodge -- A `` floater '' guard on E Block ( not permanently
assigned there ) . Jack Van Hay -- A guard who is part of the execution team . He operates the switch room . Toot - Toot -- A trustee who stands in for the
condemned during execution rehearsals and sells snacks to prisoners and guards . Burt Hammersmith -- A reporter who wrote on the Detterick twins ' murders and
John Coffey 's trial .
</item 2>

<item 3>
The Green Mile (novel)
Despite believing himself to be an `` Enlightenment '' man , he displays prejudice in his stance on `` Negroes '' and tries to convince Paul of John Coffey 's
guilt . Elaine Connelly -- A friend of Paul in the present - day nursing home where he tells his story . She is later revealed to be the grandmother of the
Speaker of the Georgia House of Representatives , using this position to scare Brad Dolan from harassing Paul . Homer Cribus -- The Sheriff of Trapingus County
, where the murders of the Detterick twins took place . Despite playing no part in apprehending John Coffey , he later shows up at the execution . An outspoken
Baptist with strong racial prejudice and immensely overweight , he later succumbs to a heart attack while having sex with a 17 - year - old African American in
his office . Rob McGee -- The Deputy Sheriff of Trapingus County , who led the search party which found John Coffey . While displaying strong doubt about Coffey
's guilt after being shown signs of innocence by Paul Edgecombe , he is still powerless to call for an appeal as he is subordinate to Sheriff Cribus , whom he
allegedly hopes to succeed .
</item 3>

<item 4>
The Green Mile (novel)
Brad Dolan -- A malicious nursing home employee who harasses Paul Edgecombe . Paul strongly compares him to Percy Wetmore , and several times mistakes him for
Percy , despite the latter dying in 1965 . Kathe and Cora Detterick -- The two young girls whom John Coffey was convicted of raping and murdering . Killed by
William Wharton . Klaus and Marjorie Detterick -- The parents of Kathe and Cora . Reception ( edit ) The Green Mile won the Bram Stoker Award for Best Novel in
1996 . In 1997 , The Green Mile was nominated as Best Novel for the British Fantasy Award and the Locus Award . In 2003 the book was listed on the BBC 's The
Big Read poll of the UK 's `` best - loved novel . '' Film adaptation ( edit ) Main article : The Green Mile ( film ) Frank Darabont adapted the novel into a
screenplay for a feature film of the same name . Released in 1999 , the film was directed by Darabont and starred Tom Hanks as Paul Edgecombe and Michael Clarke
Duncan as John Coffey .
</item 4>

<item 5>
The Green Mile (novel)
He is resurrected by John Coffey after being stomped on by Percy Wetmore . This gives him increased longevity and he finally dies 64 years later . Other
characters ( edit ) Arthur `` The President '' Flanders -- An inmate on death row , convicted of killing his father in an insurance - fraud scheme . His
sentence is commuted to life imprisonment , but he is later murdered by an unknown inmate in the prison laundry . Arlen `` The Chief '' Bitterbuck -- A Washita
Cherokee death - row inmate , convicted of killing a man in a drunken brawl over a pair of boots . His execution is the first of three mentioned in Paul 's
story . Janice Edgecombe -- Paul Edgecombe 's wife . Dies in a bus accident on the way to her grandchild 's graduation in 1956 . Hal Moores -- The warden at
Cold Mountain Penitentiary . Melinda Moores -- Warden Moores 's wife , who is dying of a brain tumor and is cured by John Coffey . She eventually dies of a
heart attack ten years later . Curtis Anderson -- The assistant warden .
</item 5>

<item 6>
The Green Mile (novel)
Paul is unsure how to help John , but John tells him not to worry , as he is ready to die anyway , wanting to escape the cruelty of the world . John 's
execution is the last one in which Paul participates . He introduces Mr. Jingles to Elaine just before the mouse dies , having lived 64 years past these events
, and explains that those healed by John gained an unnaturally long lifespan . Elaine dies shortly after , never learning how Paul 's wife died in his arms
immediately after they suffered a bus accident , and that he then saw John Coffey 's ghost watching him from an overpass . Paul seems to be all alone , now 104
years old , and wondering how much longer he will live . Characters ( edit ) Paul Edgecombe -- The protagonist and narrator of the book and the death - row
supervisor at Cold Mountain Penitentiary . He is 40 years old when the main bulk of the story takes place , in 1932 . He is a caring man and takes excellent
care of the men on his block , avoiding conflict and keeping the peace whenever possible .
</item 6>

<item 7>
The Green Mile (novel)
He is the first character to discover John Coffey 's amazing abilities , when the prisoner cures his urinary tract infection . It is also his idea to take
Coffey to try to cure Melinda , Warden Hal Moores ' wife , of her brain tumor . He transferred to Boy 's Correctional with Brutus Howell shortly after Coffey 's
execution . Brutus `` Brutal '' Howell -- He is second in command on the Green Mile and Paul 's best friend . He is a tall , imposing man but not violent at all
unless necessary . His nickname of `` Brutal '' is intended as irony . A former football tackle who had gone on to play at Louisiana State University ( LSU ) in
his youth , he eventually suffers a fatal heart attack at home , in his fifties , about twenty - five years after Coffey 's execution . John Coffey -- He is a
massive black man ( 6 ft 8 in tall ) , on death row for the alleged rape and murder of two young girls . He is very quiet and prefers to keep to himself , weeps
almost constantly , and is afraid of the dark .
</item 7>

<item 8>
The Green Mile (novel)
He is later attacked by `` Wild Bill '' Wharton , consequently wets himself , and is teased by Delacroix for it . In retaliation , Percy deliberately sabotages
Delacroix 's execution . At the end of the story he is sent to the Briar Ridge mental institution , originally considered for a job but now as a patient , after
Coffey transferred Melinda 's disease to him which caused him to kill William Wharton . He eventually lives through a hospital fire and dies in 1965 . William
`` Wild Bill '' Wharton -- The antagonist of the story . He is on death row for various crimes . He does not like the nickname `` Wild Bill '' but prefers to be
called `` Billy the Kid '' , a name which he has tattooed on his forearm . When he first arrives he manages to convince the guards that he is in a drugged
stupor , only to attack and attempt to strangle to death Dean Stanton when they reach E Block . He continues to wreak havoc on the Mile and plays tricks such as
urinating on the guards , amongst other things . He is punished by being placed in solitary confinement , but never seems to learn his lesson .
</item 8>

<item 9>
The Green Mile (novel)
Even at the end , during his execution , he asks Paul Edgecombe not to put on the traditional black silk mask used to block the view of the prisoner 's face
because he fears the dark . Coffey is described as `` knowing his own name and not much else '' and lacks the capability to do so much as tie a simple knot .
However , he is convicted of luring the girls away from their home , disposing of the watchdog , carefully planning and using abilities he would otherwise not
be expected to have . He is the calmest and mildest prisoner the guards have ever seen , despite his hulking form . He turns out to be innocent of what he is
accused of , but chooses to die anyway . Percy Wetmore -- He is a young and sadistic guard . He is disliked by Paul and the other guards because of his ways ,
but they can not do anything about it because he is the nephew of the governor 's wife . He is very homophobic and attacks Eduard Delacroix for allegedly
touching him , although it was an accident caused by Del stumbling out of the prison truck .
</item 9>

<item 10>
The Green Mile (novel)
This year marks the arrival of John Coffey , a 6 ft 8 in powerfully built black man who has been convicted of raping and murdering two small white girls .
During his time on the Mile , John interacts with fellow prisoners Eduard `` Del '' Delacroix , a Cajun arsonist , rapist , and murderer , and William Wharton (
`` Billy the Kid '' to himself , `` Wild Bill '' to the guards ) , a wild - acting and dangerous multiple murderer who is determined to make as much trouble as
he can before he is executed . Other inhabitants include Arlen Bitterbuck , a Native American convicted of killing a man in a fight over a pair of boots ( also
the first character to die in the electric chair ) ; Arthur Flanders , a real estate executive who killed his father to perpetrate insurance fraud , and whose
sentence is eventually commuted to life imprisonment ( while serving his sentence , he is killed by another inmate in the laundry room ) ; and Mr. Jingles , a
mouse , to whom Del teaches various tricks . Paul and the other guards are irritated throughout the book by Percy Wetmore , a sadistic guard who enjoys
antagonizing the prisoners .
</item 10>
=== END ITEM id=79cc17e5367ed7db ===

=== ITEM id=a71c31b1bc162b4f ===
Gold sentences and facts:
Sentence 0: The College Football Playoff selection committee chose the semifinalists following the conclusion of the 2017 regular season . Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 .
  Fact 0.0: The College Football Playoff selection committee chose the semifinalists.
  Fact 0.1: The semifinalists were chosen following the conclusion of the 2017 regular season.
  Fact 0.2: Alabama advanced to the national championship.
  Fact 0.3: Georgia advanced to the national championship.
  Fact 0.4: Alabama won its semifinal game hosted by the Sugar Bowl on January 1, 2018.
  Fact 0.5: Georgia won its semifinal game hosted by the Rose Bowl on January 1, 2018.
  Fact 0.6: The championship game was played at Mercedes-Benz Stadium in Atlanta, Georgia.
  Fact 0.7: The championship game was played on January 8, 2018.
Sentence 1: The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National Championship on November 4 , 2015 .
  Fact 1.0: Mercedes-Benz Stadium in Atlanta, Georgia was announced as the host site for the fourth College Football National Championship.
  Fact 1.1: The host site announcement was made on November 4, 2015.

Retrieved items:
<item 1>
2018 College Football Playoff National Championship
On January 8 , 2018 , the network announced that its broadcast would feature a live performance by Kendrick Lamar during halftime . This performance was
separate from the event proper at Mercedes - Benz Stadium ( which featured a traditional halftime show with the marching bands of the participating teams ) ,
and originated from Centennial Olympic Park . Dispute ( edit ) Following an undefeated season in the American Athletic Conference and their victory over the
Auburn Tigers -- who had beaten both the Alabama Crimson Tide and the Georgia Bulldogs in the regular season before losing to Georgia in the 2017 SEC
Championship Game -- in the 2018 Peach Bowl , the UCF Knights claimed to have won the national championship by virtue of being the only undefeated team in the
FBS level of college football . This claim , however , is not recognized by any major poll . The selection committee ranked UCF 12th at the end of the regular
season , below a number of two - loss teams from the Power Five conferences , which left them out of the CFP . UCF announced plans for a victory parade at
Disney World and to hang a national championship banner at their stadium to celebrate this achievement .
</item 1>

<item 2>
2018 College Football Playoff National Championship
Outback Bowl 2013 Capital One Bowl 2014 Gator Bowl 2014 Belk Bowl 2016 TaxSlayer Bowl 2016 Liberty Bowl 2018 Rose Bowl / 2018 College Football Playoff National
Championship # # denotes national championship game / denotes College Football Playoff semifinal game 2017 -- 18 NCAA Division I championships Baseball Men 's
basketball Women 's basketball Beach volleyball Bowling Cross country Fencing Field hockey Football ( FBS ) † Football ( FCS ) Men 's golf Women 's golf Men 's
gymnastics Women 's gymnastics Men 's ice hockey Women 's ice hockey Indoor track and field Men 's lacrosse Women 's lacrosse Outdoor track and field Rifle
Rowing Skiing Men 's soccer Women 's soccer Softball Men 's swimming and diving Women 's swimming and diving Tennis Men 's volleyball Women 's volleyball Men 's
water polo Women 's water polo Wrestling † Not an officially sanctioned NCAA championship Retrieved from `` https://en.wikipedia.org/w/index.
</item 2>

<item 3>
2018 College Football Playoff National Championship
28 ) Camping World ( Dec. 28 ) Alamo ( Dec. 28 ) Holiday ( Dec. 28 ) Belk ( Dec. 29 ) Cotton ( Dec. 29 ) Sun ( Dec. 29 ) Music City ( Dec. 29 ) Arizona ( Dec.
29 ) Fiesta ( Dec. 30 ) Liberty ( Dec. 30 ) Orange ( Dec. 30 ) TaxSlayer ( Dec. 30 ) Citrus ( Jan. 1 ) Outback ( Jan. 1 ) Peach ( Jan. 1 ) College Football
Playoff games : Rose ( semifinal ) ( Jan. 1 ) Sugar ( semifinal ) ( Jan. 1 ) National Championship Game ( Jan. 8 ) All - Star Games East -- West Shrine Game (
Jan. 20 ) NFLPA Collegiate Bowl ( Jan. 20 ) Senior Bowl ( Jan. 27 ) NCAA Football Championship Games Division I FBS 2007 * 2008 * 2009 * 2010 * 2011 * 2012 *
2013 * 2014 * 2015 * 2016 * 2017 * 2018 * 2019 *
</item 3>

<item 4>
2018 College Football Playoff National Championship
Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on
January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 . Contents ( hide ) 1 Background 2 2018
College Football Playoff 3 Teams 3.1 Alabama 3.2 Georgia 4 Game summary 4.1 Scoring summary 4.2 Statistics 5 Broadcasting 6 Dispute 7 See also 8 References 9
External links Background ( edit ) The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National
Championship on November 4 , 2015 .
</item 4>

<item 5>
2018 College Football Playoff National Championship
Jump up ^ Alex Scarbrough ( November 4 , 2015 ) . `` Atlanta , Santa Clara and New Orleans land CFP title games for 2018 -- 20 '' . ESPN.com . Retrieved
November 4 , 2015 . ^ Jump up to : `` Selection Committee Rankings : Final Top 25 Rankings '' ( PDF ) . College Football Playoff . December 3 , 2017 . Retrieved
December 3 , 2017 . Jump up ^ `` Alabama 's Sugar Bowl win against Clemson sets up all - SEC final : Final score , stats , analysis '' . nola.com . January 1 ,
2018 . Retrieved January 2 , 2018 . Jump up ^ Emerson , Seth ( January 1 , 2018 ) . `` Georgia to play for national championship after Rose Bowl win in 2 OT ''
. ajc.com . Retrieved January 2 , 2018 . ^ Jump up to : Hoffman , Benjamin ; Drape , Joe ; Tracy , Marc ( January 8 , 2018 ) . `` National Championship Game :
Georgia vs.
</item 5>

<item 6>
2018 College Football Playoff National Championship
champions Mythical national championship Plus - One system Power Five conferences Championship games for each season are played in January , while semifinal
games are identified by season year , alternating between December and January Links to related articles 2017 -- 18 NCAA football bowl game season New Mexico (
Dec. 16 ) Las Vegas ( Dec. 16 ) Camellia ( Dec. 16 ) Cure ( Dec. 16 ) New Orleans ( Dec. 16 ) Boca Raton ( Dec. 19 ) Frisco ( Dec. 20 ) Gasparilla ( Dec. 21 )
Famous Idaho Potato ( Dec. 22 ) Bahamas ( Dec. 22 ) Armed Forces ( Dec. 23 ) Birmingham ( Dec. 23 ) Dollar General ( Dec. 23 ) Hawaii ( Dec. 24 ) Quick Lane (
Dec. 26 ) Heart of Dallas ( Dec. 26 ) Cactus ( Dec. 26 ) Independence ( Dec. 27 ) Pinstripe ( Dec. 27 ) Texas ( Dec. 27 ) Foster Farms ( Dec. 27 ) Military (
Dec. 28 ) Camping World ( Dec. 28 ) Alamo ( Dec.
</item 6>

<item 7>
2018 College Football Playoff National Championship
Alabama Defensive : # 94 DT Da'Ron Payne , Jr . Alabama Favorite Alabama by 3.5 National anthem Zac Brown Band Referee Dan Capron ( Big Ten ) Halftime show
Georgia Redcoat Marching Band Million Dollar Band Attendance 77,430 United States TV coverage Network ESPN / ESPN Radio / ESPN Deportes Announcers Chris Fowler
, Kirk Herbstreit , Maria Taylor and Tom Rinaldi ( ESPN ) Sean McDonough , Todd Blackledge , Holly Rowe and Ian Fitzsimmons ( ESPN Radio ) College Football
Playoff National Championship < 2017 2019 > The 2018 College Football Playoff National Championship was a college football bowl game that determined the
national champion in the NCAA Division I Football Bowl Subdivision for the 2017 season . The Alabama Crimson Tide defeated the Georgia Bulldogs 26 -- 23 in
overtime . Alabama overcame a 13 -- 0 deficit at halftime . Tua Tagovailoa and Da'Ron Payne were respectively named the offensive and defensive players of the
game . The College Football Playoff selection committee chose the semifinalists following the conclusion of the 2017 regular season .
</item 7>

<item 8>
2018 College Football Playoff National Championship
2018 College football Playoff national championship - wikipedia 2018 College football Playoff national championship Jump to : navigation , search 2018 College
Football Playoff National Championship Alabama Crimson Tide Georgia Bulldogs ( 12 -- 1 ) ( 13 -- 1 ) SEC SEC 26 23 Head coach : Nick Saban Head coach : Kirby
Smart AP Coaches CFP AP Coaches CFP OT Total Alabama 0 0 10 10 6 26 Georgia 0 13 7 0 23 Date January 8 , 2018 Season 2017 Stadium Mercedes - Benz Stadium
Location Atlanta , Georgia MVP Offensive : # 13 QB Tua Tagovailoa , Fr . Alabama Defensive : # 94 DT Da'Ron Payne , Jr .
</item 8>

<item 9>
2018 College Football Playoff National Championship
2018 College football Playoff ( edit ) Semifinals Championship January 1 -- Sugar Bowl Mercedes - Benz Superdome , New Orleans 1 Clemson 6 4 Alabama 24 January
8 -- National Championship Mercedes - Benz Stadium , Atlanta 4 Alabama ( OT ) 26 January 1 -- Rose Bowl Rose Bowl , Pasadena 3 Georgia 23 2 Oklahoma 48 3
Georgia ( 2OT ) 54 This bracket : view talk edit The College Football Playoff ( CFP ) selected four teams to advance to the playoff : the Clemson Tigers ,
Oklahoma Sooners , Georgia Bulldogs , and Alabama Crimson Tide . Teams ( edit ) Alabama ( edit ) See also : 2017 Alabama Crimson Tide football team The Alabama
Crimson Tide defeated the Clemson Tigers in the 2018 Sugar Bowl by a score of 24 -- 6 to reach the championship game . The Tide had a 12 -- 1 season , becoming
the SEC West Division co-champions with the Auburn Tigers . The Crimson Tide were coached by Nick Saban .
</item 9>

<item 10>
2018 College Football Playoff National Championship
1990 Sugar Bowl 1991 Fiesta Bowl 1991 Blockbuster Bowl 1993 Sugar Bowl # 1993 Gator Bowl 1995 Florida Citrus Bowl 1997 Outback Bowl 1998 Music City Bowl 2000
Orange Bowl 2001 Independence Bowl 2004 Music City Bowl 2006 Cotton Bowl Classic 2006 Independence Bowl 2007 Independence Bowl 2009 Sugar Bowl 2010 BCS National
Championship Game # 2011 Capital One Bowl 2012 BCS National Championship Game # 2013 BCS National Championship Game # 2014 Sugar Bowl 2015 Sugar Bowl / 2015
Cotton Bowl Classic ( Dec ) / 2016 College Football Playoff National Championship # 2016 Peach Bowl / 2017 College Football Playoff National Championship # 2018
Sugar Bowl / 2018 College Football Playoff National Championship # # denotes national championship game / denotes College Football Playoff semifinal game
Georgia Bulldogs bowl games 1942 Orange Bowl 1943 Rose Bowl 1946 Oil Bowl 1947 Sugar Bowl 1948 Gator Bowl 1949 Orange Bowl 1950 Presidential Cup 1960 Orange
Bowl 1964 Sun Bowl 1966 Cotton Bowl Classic
</item 10>
=== END ITEM id=a71c31b1bc162b4f ===
